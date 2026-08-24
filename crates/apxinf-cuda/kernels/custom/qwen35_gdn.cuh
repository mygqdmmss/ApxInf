#pragma once

// Correctness-first Qwen3.5 gated delta-network primitives.
//
// The convolution ring is BF16, matching the checkpoint activation path. The
// recurrent matrix is deliberately FP32: its long-lived request state must not
// accumulate BF16 roundoff. Launch adapters keep the current/scratch buffers
// separate so Rust can commit a step only after checking the device status.

__device__ __forceinline__ float qwen35_gdn_sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float qwen35_gdn_silu(float value) {
  return value * qwen35_gdn_sigmoid(value);
}

__device__ __forceinline__ float qwen35_gdn_softplus(float value) {
  if (value > 20.0f) return value;
  if (value < -20.0f) return expf(value);
  return log1pf(expf(value));
}

// Transformers' Qwen3.5 path applies l2norm while q/k are still BF16.  Keep
// both the reciprocal norm and each normalized element at that boundary before
// converting the vectors to the FP32 recurrent computation.
__device__ __forceinline__ __nv_bfloat16 qwen35_gdn_bf16_add(
    __nv_bfloat16 lhs, __nv_bfloat16 rhs) {
  return __float2bfloat16(__bfloat162float(lhs) + __bfloat162float(rhs));
}

__device__ __forceinline__ __nv_bfloat16 qwen35_gdn_bf16_square(
    __nv_bfloat16 value) {
  const float scalar = __bfloat162float(value);
  return __float2bfloat16(scalar * scalar);
}

__device__ __forceinline__ float qwen35_gdn_bf16_l2_scale(
    __nv_bfloat16 sum) {
  const __nv_bfloat16 eps_bf16 = __float2bfloat16(1e-6f);
  const __nv_bfloat16 denominator = qwen35_gdn_bf16_add(sum, eps_bf16);
  // The pinned pure Transformers implementation applies rsqrt to the BF16
  // reduction for every row. It has no sequence-length-dependent tail path.
  return __bfloat162float(
      __float2bfloat16(__frsqrt_rn(__bfloat162float(denominator))));
}

__device__ __forceinline__ float qwen35_gdn_bf16_l2_value(
    float value, float inverse_norm) {
  return __bfloat162float(__float2bfloat16(value * inverse_norm));
}

__global__ void qwen35_gdn_check_finite_bf16_kernel(
    const __nv_bfloat16* input, uint32_t* error_flags, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  if (!isfinite(__bfloat162float(input[index]))) atomicOr(error_flags, 1U);
}

__global__ void qwen35_gdn_conv_bf16_kernel(
    const __nv_bfloat16* ring_in, __nv_bfloat16* ring_out,
    const __nv_bfloat16* input, const __nv_bfloat16* weights,
    __nv_bfloat16* output, uint32_t* error_flags, int channels,
    int kernel, int cursor) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;

  const int64_t base = static_cast<int64_t>(channel) * kernel;
  for (int slot = 0; slot < kernel; ++slot) {
    ring_out[base + slot] = ring_in[base + slot];
  }

  const float current = __bfloat162float(input[channel]);
  if (!isfinite(current)) atomicOr(error_flags, 1U);
  ring_out[base + cursor] = __float2bfloat16(current);

  float sum = 0.0f;
  for (int offset = 0; offset < kernel; ++offset) {
    const float value =
        __bfloat162float(ring_out[base + ((cursor + 1 + offset) % kernel)]);
    const float weight = __bfloat162float(weights[base + offset]);
    if (!isfinite(weight) || !isfinite(value)) atomicOr(error_flags, 1U);
    sum += value * weight;
  }
  const float activated = qwen35_gdn_silu(sum);
  if (!isfinite(sum) || !isfinite(activated)) atomicOr(error_flags, 2U);
  output[channel] = __float2bfloat16(activated);
  if (!isfinite(__bfloat162float(output[channel]))) atomicOr(error_flags, 2U);
}

// Sequence prefill follows the causal_conv1d semantics used by the
// Transformers reference: each row sees only the current row and the prior
// kernel-1 rows, with zero-filled left padding. One thread owns one channel
// so the ring update is ordered in time without cross-channel synchronization.
__global__ void qwen35_gdn_conv_prefill_bf16_kernel(
    const __nv_bfloat16* ring_in, __nv_bfloat16* ring_out,
    const __nv_bfloat16* input, const __nv_bfloat16* weights,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int channels,
    int kernel, int cursor) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;

  const int64_t base = static_cast<int64_t>(channel) * kernel;
  for (int slot = 0; slot < kernel; ++slot) {
    ring_out[base + slot] = ring_in[base + slot];
  }

  int local_cursor = cursor;
  for (int row = 0; row < rows; ++row) {
    const int64_t input_index = static_cast<int64_t>(row) * channels + channel;
    const float current = __bfloat162float(input[input_index]);
    if (!isfinite(current)) atomicOr(error_flags, 1U);
    ring_out[base + local_cursor] = __float2bfloat16(current);

    float sum = 0.0f;
    for (int offset = 0; offset < kernel; ++offset) {
      const float value = __bfloat162float(
          ring_out[base + ((local_cursor + 1 + offset) % kernel)]);
      const float weight = __bfloat162float(weights[base + offset]);
      if (!isfinite(weight) || !isfinite(value)) atomicOr(error_flags, 1U);
      sum += value * weight;
    }
    const float activated = qwen35_gdn_silu(sum);
    if (!isfinite(sum) || !isfinite(activated)) atomicOr(error_flags, 2U);
    const int64_t output_index = static_cast<int64_t>(row) * channels + channel;
    output[output_index] = __float2bfloat16(activated);
    if (!isfinite(__bfloat162float(output[output_index]))) {
      atomicOr(error_flags, 2U);
    }
    local_cursor = (local_cursor + 1) % kernel;
  }
}

__global__ void qwen35_gdn_recurrent_bf16_f32_kernel(
    const float* state_in, float* state_out, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    __nv_bfloat16* output, uint32_t* error_flags, int key_heads,
    int value_heads, int key_dim, int value_dim) {
  const int head = blockIdx.x;
  if (head >= value_heads) return;
  const int repeat_factor = value_heads / key_heads;
  const int key_head = head / repeat_factor;

  __shared__ float query_inverse_norm;
  __shared__ float key_inverse_norm;
  if (threadIdx.x == 0) {
    __nv_bfloat16 query_sum = __float2bfloat16(0.0f);
    __nv_bfloat16 key_sum = __float2bfloat16(0.0f);
    const int64_t head_base = static_cast<int64_t>(key_head) * key_dim;
    for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
      const __nv_bfloat16 query_bf16 = query[head_base + key_dimension];
      const __nv_bfloat16 key_bf16 = key[head_base + key_dimension];
      const float query_value = __bfloat162float(query_bf16);
      const float key_value = __bfloat162float(key_bf16);
      if (!isfinite(query_value) || !isfinite(key_value)) {
        atomicOr(error_flags, 1U);
      }
      query_sum = qwen35_gdn_bf16_add(query_sum,
                                      qwen35_gdn_bf16_square(query_bf16));
      key_sum = qwen35_gdn_bf16_add(key_sum,
                                    qwen35_gdn_bf16_square(key_bf16));
    }
    query_inverse_norm = qwen35_gdn_bf16_l2_scale(query_sum);
    key_inverse_norm = qwen35_gdn_bf16_l2_scale(key_sum);
    if (!isfinite(query_inverse_norm) || !isfinite(key_inverse_norm)) {
      atomicOr(error_flags, 2U);
    }
  }
  __syncthreads();

  const float a_value = __bfloat162float(a[head]);
  const float b_value = __bfloat162float(b[head]);
  const float a_log_value = __bfloat162float(a_log[head]);
  const float dt_bias_value = __bfloat162float(dt_bias[head]);
  if (!isfinite(a_value) || !isfinite(b_value) || !isfinite(a_log_value) ||
      !isfinite(dt_bias_value)) {
    atomicOr(error_flags, 1U);
  }
  const float decay_log =
      -expf(a_log_value) * qwen35_gdn_softplus(a_value + dt_bias_value);
  const float decay = expf(decay_log);
  const float beta = __bfloat162float(
      __float2bfloat16(qwen35_gdn_sigmoid(b_value)));
  if (!isfinite(decay) || !isfinite(beta)) atomicOr(error_flags, 2U);

  const int64_t state_base =
      static_cast<int64_t>(head) * key_dim * value_dim;
  const int64_t value_base = static_cast<int64_t>(head) * value_dim;
  const int64_t query_base = static_cast<int64_t>(key_head) * key_dim;
  const float query_scale = rsqrtf(static_cast<float>(key_dim));

  for (int value_dimension = threadIdx.x; value_dimension < value_dim;
       value_dimension += blockDim.x) {
    float memory = 0.0f;
    for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
      const int64_t state_index =
          state_base + static_cast<int64_t>(key_dimension) * value_dim +
          value_dimension;
      const float old_value = state_in[state_index];
      const float decayed = old_value * decay;
      state_out[state_index] = decayed;
      const float key_value =
          qwen35_gdn_bf16_l2_value(
              __bfloat162float(key[query_base + key_dimension]),
              key_inverse_norm);
      if (!isfinite(old_value) || !isfinite(decayed) || !isfinite(key_value)) {
        atomicOr(error_flags, 1U);
      }
      memory += decayed * key_value;
    }

    const float value_value =
        __bfloat162float(value[value_base + value_dimension]);
    const float delta = (value_value - memory) * beta;
    if (!isfinite(value_value) || !isfinite(memory) || !isfinite(delta)) {
      atomicOr(error_flags, 2U);
    }
    float result = 0.0f;
    for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
      const int64_t state_index =
          state_base + static_cast<int64_t>(key_dimension) * value_dim +
          value_dimension;
      const float key_value =
          qwen35_gdn_bf16_l2_value(
              __bfloat162float(key[query_base + key_dimension]),
              key_inverse_norm);
      const float updated = state_out[state_index] + key_value * delta;
      state_out[state_index] = updated;
      const float query_value =
          qwen35_gdn_bf16_l2_value(
              __bfloat162float(query[query_base + key_dimension]),
              query_inverse_norm);
      result += updated * query_value * query_scale;
      if (!isfinite(updated) || !isfinite(query_value) || !isfinite(result)) {
        atomicOr(error_flags, 2U);
      }
    }
    output[value_base + value_dimension] = __float2bfloat16(result);
    if (!isfinite(__bfloat162float(output[value_base + value_dimension]))) {
      atomicOr(error_flags, 2U);
    }
  }
}

// Chunked sequence update matching torch_chunk_gated_delta_rule.  A block owns
// one value head. The implementation intentionally keeps the chunk algebra
// explicit (triangular inverse, transformed values, cumulative-decayed keys,
// and one state transition per 64-token chunk); the per-head workspace is
// supplied by the Rust wrapper so the long-lived state remains FP32.
constexpr int QWEN35_GDN_CHUNK_SIZE = 64;

__global__ void qwen35_gdn_sequence_recurrent_bf16_f32_kernel(
    const float* state_in, float* state_out, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int key_heads,
    int value_heads, int key_dim, int value_dim, float* workspace,
    int64_t workspace_stride) {
  const int head = blockIdx.x;
  if (head >= value_heads) return;
  // The first version is deliberately correctness-first: one lane performs
  // the ordered scalar algebra while the block owns a disjoint head/state and
  // workspace slice. This avoids cross-head synchronization hazards and keeps
  // all intermediate operations in the same order as the reference.
  if (threadIdx.x != 0) return;
  const int repeat_factor = value_heads / key_heads;
  const int key_head = head / repeat_factor;
  const int64_t state_base = static_cast<int64_t>(head) * key_dim * value_dim;
  const int64_t value_width = static_cast<int64_t>(value_heads) * value_dim;
  const int64_t query_width = static_cast<int64_t>(key_heads) * key_dim;

  const int64_t q_count = static_cast<int64_t>(QWEN35_GDN_CHUNK_SIZE) * key_dim;
  const int64_t v_count = static_cast<int64_t>(QWEN35_GDN_CHUNK_SIZE) * value_dim;
  float* block_workspace = workspace + static_cast<int64_t>(head) * workspace_stride;
  float* q_norm = block_workspace;
  float* k_norm = q_norm + q_count;
  float* values = k_norm + q_count;
  float* beta = values + v_count;
  float* g_cum = beta + QWEN35_GDN_CHUNK_SIZE;
  float* attn = g_cum + QWEN35_GDN_CHUNK_SIZE;
  float* transformed_values = attn +
      static_cast<int64_t>(QWEN35_GDN_CHUNK_SIZE) * QWEN35_GDN_CHUNK_SIZE;
  float* k_cumdecay = transformed_values + v_count;
  float* v_new = k_cumdecay + q_count;

  for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
    for (int value_dimension = 0; value_dimension < value_dim;
         ++value_dimension) {
      const int64_t index = state_base +
          static_cast<int64_t>(key_dimension) * value_dim + value_dimension;
      const float initial = state_in[index];
      state_out[index] = initial;
      if (!isfinite(initial)) atomicOr(error_flags, 1U);
    }
  }

  const int chunk_count =
      (rows + QWEN35_GDN_CHUNK_SIZE - 1) / QWEN35_GDN_CHUNK_SIZE;
  const float query_scale = rsqrtf(static_cast<float>(key_dim));
  const float a_log_value = __bfloat162float(a_log[head]);
  const float dt_bias_value = __bfloat162float(dt_bias[head]);
  if (!isfinite(a_log_value) || !isfinite(dt_bias_value) ||
      !isfinite(query_scale)) {
    atomicOr(error_flags, 1U);
  }

  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    // Load, normalize, and materialize the BF16 boundary for one chunk.
    for (int token = 0; token < QWEN35_GDN_CHUNK_SIZE; ++token) {
      const int row = chunk * QWEN35_GDN_CHUNK_SIZE + token;
      const bool valid = row < rows;
      const int64_t q_base = static_cast<int64_t>(row) * query_width +
          static_cast<int64_t>(key_head) * key_dim;
      const int64_t value_base = static_cast<int64_t>(row) * value_width +
          static_cast<int64_t>(head) * value_dim;
      __nv_bfloat16 query_sum = __float2bfloat16(0.0f);
      __nv_bfloat16 key_sum = __float2bfloat16(0.0f);
      for (int key_dimension = 0; key_dimension < key_dim;
           ++key_dimension) {
        const __nv_bfloat16 query_bf16 = valid
            ? query[q_base + key_dimension] : __float2bfloat16(0.0f);
        const __nv_bfloat16 key_bf16 = valid
            ? key[q_base + key_dimension] : __float2bfloat16(0.0f);
        const float query_value = __bfloat162float(query_bf16);
        const float key_value = __bfloat162float(key_bf16);
        if (!isfinite(query_value) || !isfinite(key_value)) {
          atomicOr(error_flags, 1U);
        }
        query_sum = qwen35_gdn_bf16_add(
            query_sum, qwen35_gdn_bf16_square(query_bf16));
        key_sum = qwen35_gdn_bf16_add(
            key_sum, qwen35_gdn_bf16_square(key_bf16));
      }
      const float query_inverse_norm =
          qwen35_gdn_bf16_l2_scale(query_sum);
      const float key_inverse_norm =
          qwen35_gdn_bf16_l2_scale(key_sum);
      if (!isfinite(query_inverse_norm) || !isfinite(key_inverse_norm)) {
        atomicOr(error_flags, 2U);
      }
      for (int key_dimension = 0; key_dimension < key_dim;
           ++key_dimension) {
        const __nv_bfloat16 query_bf16 = valid
            ? query[q_base + key_dimension] : __float2bfloat16(0.0f);
        const __nv_bfloat16 key_bf16 = valid
            ? key[q_base + key_dimension] : __float2bfloat16(0.0f);
        q_norm[static_cast<int64_t>(token) * key_dim + key_dimension] =
            qwen35_gdn_bf16_l2_value(__bfloat162float(query_bf16),
                                     query_inverse_norm) * query_scale;
        k_norm[static_cast<int64_t>(token) * key_dim + key_dimension] =
            qwen35_gdn_bf16_l2_value(__bfloat162float(key_bf16),
                                     key_inverse_norm);
      }
      for (int value_dimension = 0; value_dimension < value_dim;
           ++value_dimension) {
        const __nv_bfloat16 value_bf16 = valid
            ? value[value_base + value_dimension] : __float2bfloat16(0.0f);
        const float value_float = __bfloat162float(value_bf16);
        if (!isfinite(value_float)) atomicOr(error_flags, 1U);
        values[static_cast<int64_t>(token) * value_dim + value_dimension] =
            value_float;
      }
      float g_value = 0.0f;
      float beta_value = 0.0f;
      if (valid) {
        const int64_t gate_index = static_cast<int64_t>(row) * value_heads + head;
        const float a_value = __bfloat162float(a[gate_index]);
        const float b_value = __bfloat162float(b[gate_index]);
        if (!isfinite(a_value) || !isfinite(b_value)) {
          atomicOr(error_flags, 1U);
        }
        g_value = -expf(a_log_value) *
            qwen35_gdn_softplus(a_value + dt_bias_value);
        beta_value = __bfloat162float(
            __float2bfloat16(qwen35_gdn_sigmoid(b_value)));
      }
      if (!isfinite(g_value) || !isfinite(beta_value)) {
        atomicOr(error_flags, 2U);
      }
      beta[token] = beta_value;
      g_cum[token] = g_value;
      if (token > 0) g_cum[token] += g_cum[token - 1];
      if (!isfinite(g_cum[token])) atomicOr(error_flags, 2U);
    }

    // Build the strictly-lower chunk transition and solve its triangular
    // inverse row by row, exactly as the reference implementation does.
    for (int i = 0; i < QWEN35_GDN_CHUNK_SIZE; ++i) {
      for (int j = 0; j < QWEN35_GDN_CHUNK_SIZE; ++j) {
        float entry = 0.0f;
        if (j < i) {
          float dot = 0.0f;
          for (int key_dimension = 0; key_dimension < key_dim;
               ++key_dimension) {
            const float kb = k_norm[static_cast<int64_t>(i) * key_dim +
                                    key_dimension] * beta[i];
            const float kj = k_norm[static_cast<int64_t>(j) * key_dim +
                                    key_dimension];
            dot += kb * kj;
          }
          entry = -dot * expf(g_cum[i] - g_cum[j]);
        }
        attn[static_cast<int64_t>(i) * QWEN35_GDN_CHUNK_SIZE + j] = entry;
      }
    }
    for (int i = 1; i < QWEN35_GDN_CHUNK_SIZE; ++i) {
      float row_values[QWEN35_GDN_CHUNK_SIZE];
      for (int j = 0; j < i; ++j) {
        row_values[j] = attn[static_cast<int64_t>(i) * QWEN35_GDN_CHUNK_SIZE + j];
      }
      for (int column = 0; column < i; ++column) {
        float correction = 0.0f;
        for (int j = 0; j < i; ++j) {
          correction += row_values[j] *
              attn[static_cast<int64_t>(j) * QWEN35_GDN_CHUNK_SIZE + column];
        }
        attn[static_cast<int64_t>(i) * QWEN35_GDN_CHUNK_SIZE + column] =
            row_values[column] + correction;
      }
    }
    for (int i = 0; i < QWEN35_GDN_CHUNK_SIZE; ++i) {
      attn[static_cast<int64_t>(i) * QWEN35_GDN_CHUNK_SIZE + i] = 1.0f;
    }

    for (int i = 0; i < QWEN35_GDN_CHUNK_SIZE; ++i) {
      for (int value_dimension = 0; value_dimension < value_dim;
           ++value_dimension) {
        float transformed = 0.0f;
        for (int j = 0; j < QWEN35_GDN_CHUNK_SIZE; ++j) {
          transformed += attn[static_cast<int64_t>(i) * QWEN35_GDN_CHUNK_SIZE + j] *
              values[static_cast<int64_t>(j) * value_dim + value_dimension] * beta[j];
        }
        transformed_values[static_cast<int64_t>(i) * value_dim + value_dimension] =
            transformed;
      }
      for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
        float cumulative = 0.0f;
        for (int j = 0; j < QWEN35_GDN_CHUNK_SIZE; ++j) {
          cumulative += attn[static_cast<int64_t>(i) * QWEN35_GDN_CHUNK_SIZE + j] *
              k_norm[static_cast<int64_t>(j) * key_dim + key_dimension] * beta[j] *
              expf(g_cum[j]);
        }
        k_cumdecay[static_cast<int64_t>(i) * key_dim + key_dimension] = cumulative;
      }
    }

    // Compute all per-token corrections from the state entering this chunk;
    // only after those outputs are available is the state transitioned once.
    for (int i = 0; i < QWEN35_GDN_CHUNK_SIZE; ++i) {
      const int row = chunk * QWEN35_GDN_CHUNK_SIZE + i;
      for (int value_dimension = 0; value_dimension < value_dim;
           ++value_dimension) {
        float v_prime = 0.0f;
        for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
          const int64_t state_index = state_base +
              static_cast<int64_t>(key_dimension) * value_dim + value_dimension;
          v_prime += k_cumdecay[static_cast<int64_t>(i) * key_dim + key_dimension] *
              state_out[state_index];
        }
        const float corrected = transformed_values[
            static_cast<int64_t>(i) * value_dim + value_dimension] - v_prime;
        v_new[static_cast<int64_t>(i) * value_dim + value_dimension] = corrected;
        if (!isfinite(corrected)) atomicOr(error_flags, 2U);
      }
      if (row < rows) {
        const int64_t output_base = static_cast<int64_t>(row) * value_width +
            static_cast<int64_t>(head) * value_dim;
        const float row_decay = expf(g_cum[i]);
        for (int value_dimension = 0; value_dimension < value_dim;
             ++value_dimension) {
          float interaction = 0.0f;
          float causal = 0.0f;
          for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
            const int64_t state_index = state_base +
                static_cast<int64_t>(key_dimension) * value_dim + value_dimension;
            interaction += q_norm[static_cast<int64_t>(i) * key_dim + key_dimension] *
                row_decay * state_out[state_index];
          }
          for (int j = 0; j <= i; ++j) {
            float score = 0.0f;
            for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
              score += q_norm[static_cast<int64_t>(i) * key_dim + key_dimension] *
                  k_norm[static_cast<int64_t>(j) * key_dim + key_dimension];
            }
            score *= expf(g_cum[i] - g_cum[j]);
            causal += score * v_new[static_cast<int64_t>(j) * value_dim + value_dimension];
          }
          const float result = interaction + causal;
          if (!isfinite(result)) atomicOr(error_flags, 2U);
          output[output_base + value_dimension] = __float2bfloat16(result);
          if (!isfinite(__bfloat162float(output[output_base + value_dimension]))) {
            atomicOr(error_flags, 2U);
          }
        }
      }
    }

    const float final_decay = expf(g_cum[QWEN35_GDN_CHUNK_SIZE - 1]);
    if (!isfinite(final_decay)) atomicOr(error_flags, 2U);
    for (int key_dimension = 0; key_dimension < key_dim; ++key_dimension) {
      for (int value_dimension = 0; value_dimension < value_dim;
           ++value_dimension) {
        const int64_t state_index = state_base +
            static_cast<int64_t>(key_dimension) * value_dim + value_dimension;
        float updated = state_out[state_index] * final_decay;
        for (int i = 0; i < QWEN35_GDN_CHUNK_SIZE; ++i) {
          updated += k_norm[static_cast<int64_t>(i) * key_dim + key_dimension] *
              expf(g_cum[QWEN35_GDN_CHUNK_SIZE - 1] - g_cum[i]) *
              v_new[static_cast<int64_t>(i) * value_dim + value_dimension];
        }
        state_out[state_index] = updated;
        if (!isfinite(updated)) atomicOr(error_flags, 2U);
      }
    }
  }
}

__global__ void qwen35_gdn_gated_rms_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output, uint32_t* error_flags,
    int rows, int heads, int head_dim, float eps) {
  const int row_head = blockIdx.x;
  const int total = rows * heads;
  if (row_head >= total) return;
  const int head = row_head % heads;
  const int64_t base =
      (static_cast<int64_t>(row_head) * head_dim);

  __shared__ float partial[256];
  float sum = 0.0f;
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    const float value = __bfloat162float(input[base + dimension]);
    if (!isfinite(value)) atomicOr(error_flags, 1U);
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  if (!isfinite(inverse_rms)) atomicOr(error_flags, 2U);
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    const float value = __bfloat162float(input[base + dimension]);
    const float gate_value = __bfloat162float(gate[base + dimension]);
    const float norm_weight = __bfloat162float(weight[dimension % head_dim]);
    const float result =
        value * inverse_rms * norm_weight * qwen35_gdn_silu(gate_value);
    if (!isfinite(gate_value) || !isfinite(norm_weight) || !isfinite(result)) {
      atomicOr(error_flags, 2U);
    }
    output[base + dimension] = __float2bfloat16(result);
    if (!isfinite(__bfloat162float(output[base + dimension]))) {
      atomicOr(error_flags, 2U);
    }
  }
}
