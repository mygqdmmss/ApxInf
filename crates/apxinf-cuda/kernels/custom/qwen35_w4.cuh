#pragma once

// Correctness-first asymmetric W4 projection. Weights pack eight K nibbles
// per uint32; zero-points pack eight N nibbles per uint32 for each K group.
// One block computes one [row, out] scalar with fixed 256-float shared scratch.
__global__ void qwen35_w4_project_bf16_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  __shared__ float partial[256];
  const int output_index = blockIdx.x;
  const int row = output_index / out_features;
  const int out = output_index - row * out_features;
  const int groups = (in_features + group_size - 1) / group_size;
  const int packed_k_columns = (in_features + 7) / 8;
  float sum = 0.0f;
  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    const int group = k / group_size;
    const uint32_t packed_weight =
        weight_packed[static_cast<int64_t>(out) * packed_k_columns + k / 8];
    const float quantized =
        static_cast<float>((packed_weight >> (4 * (k & 7))) & 0x0fU);
    const uint32_t packed_zero =
        zero_points[static_cast<int64_t>(out / 8) * groups + group];
    const float zero_point =
        static_cast<float>((packed_zero >> (4 * (out & 7))) & 0x0fU);
    const float scale =
        __bfloat162float(scales[static_cast<int64_t>(out) * groups + group]);
    if (!isfinite(scale)) atomicOr(error_flags, 1U);
    const float value =
        __bfloat162float(activation[static_cast<int64_t>(row) * in_features + k]);
    sum += value * (quantized - zero_point) * scale;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (!isfinite(partial[0])) atomicOr(error_flags, 2U);
    output[output_index] = __float2bfloat16(partial[0]);
  }
}
