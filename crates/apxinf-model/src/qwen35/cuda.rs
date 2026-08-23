use apxinf_core::Backend;
use apxinf_core::{DType, Device, KvCache, Shape, Tensor};
use apxinf_cuda::kernels::qwen35_w4::{Qwen35W4DeviceProjection, Qwen35W4Layout};
use apxinf_cuda::CudaContext;
use half::bf16;

use super::loader::Qwen35CheckpointInventory;

/// Device-owned W4 projection loaded from one checkpoint prefix.
pub struct Qwen35CheckpointProjection {
    layout: Qwen35W4Layout,
    projection: Qwen35W4DeviceProjection,
}

/// Device-owned weights for one real full-attention layer.
///
/// The packed projections remain in their checkpoint layout on device. Norm
/// vectors are uploaded as BF16 tensors without creating an expanded model
/// copy. This is a weight bundle only; execution is added by the model owner.
pub struct Qwen35CudaFullAttentionLayer {
    layer_index: usize,
    device_id: usize,
    q_proj: Qwen35CheckpointProjection,
    k_proj: Qwen35CheckpointProjection,
    v_proj: Qwen35CheckpointProjection,
    o_proj: Qwen35CheckpointProjection,
    gate_proj: Qwen35CheckpointProjection,
    up_proj: Qwen35CheckpointProjection,
    down_proj: Qwen35CheckpointProjection,
    input_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attention_norm: Tensor,
    n_query_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rms_epsilon: f32,
}

/// Request-local state for one CUDA full-attention layer.
///
/// The cache is deliberately one-layer and BF16-only. The complete model
/// executor will own one such cache per full-attention layer; this standalone
/// state keeps the layer smoke and its position semantics independently
/// testable without introducing a CPU mirror.
pub struct Qwen35CudaFullAttentionState {
    cache: apxinf_cuda::CudaKVCache,
    max_seq_len: usize,
    device_id: usize,
}

impl Qwen35CudaFullAttentionState {
    pub fn new(backend: &apxinf_cuda::CudaBackend, max_seq_len: usize) -> Result<Self, String> {
        if max_seq_len == 0 {
            return Err("full-attention max sequence length must be non-zero".into());
        }
        let device_id = backend.device_id();
        let cache = apxinf_cuda::CudaKVCache::new_bf16(device_id, 1, 4, 256, max_seq_len)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            cache,
            max_seq_len,
            device_id,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.cache.seq_len()
    }

    pub const fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub const fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn clear(&mut self) -> Result<(), String> {
        apxinf_core::KvCache::clear(&mut self.cache).map_err(|error| error.to_string())
    }
}

pub struct Qwen35AttentionProjectionTensors {
    pub q_gate: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

impl Qwen35CudaFullAttentionLayer {
    pub fn from_inventory(
        ctx: &CudaContext,
        inventory: &Qwen35CheckpointInventory,
        layer_index: usize,
    ) -> Result<Self, String> {
        if inventory.config.layer_types.get(layer_index)
            != Some(&super::config::LayerType::FullAttention)
        {
            return Err(format!("layer {layer_index} is not full-attention"));
        }
        let prefix = format!("model.language_model.layers.{layer_index}");
        let projection = |name: &str| {
            Qwen35CheckpointProjection::from_inventory(ctx, inventory, &format!("{prefix}.{name}"))
        };
        let norm = |name: &str, width: usize| -> Result<Tensor, String> {
            let values = inventory
                .read_tensor_bf16_values(&format!("{prefix}.{name}"))
                .map_err(|error| error.to_string())?;
            if values.len() != width {
                return Err(format!(
                    "{prefix}.{name} has {} values, expected {width}",
                    values.len()
                ));
            }
            let host = Tensor::from_bf16(Shape::new(vec![width]), &values)
                .map_err(|error| error.to_string())?;
            apxinf_cuda::transfers::to_cuda(&host, ctx.device_id())
                .map_err(|error| error.to_string())
        };

        Ok(Self {
            layer_index,
            device_id: ctx.device_id(),
            q_proj: projection("self_attn.q_proj")?,
            k_proj: projection("self_attn.k_proj")?,
            v_proj: projection("self_attn.v_proj")?,
            o_proj: projection("self_attn.o_proj")?,
            gate_proj: projection("mlp.gate_proj")?,
            up_proj: projection("mlp.up_proj")?,
            down_proj: projection("mlp.down_proj")?,
            input_norm: norm("input_layernorm.weight", inventory.config.hidden_size)?,
            q_norm: norm(
                "self_attn.q_norm.weight",
                inventory.config.full_attention_head_dim,
            )?,
            k_norm: norm(
                "self_attn.k_norm.weight",
                inventory.config.full_attention_head_dim,
            )?,
            post_attention_norm: norm(
                "post_attention_layernorm.weight",
                inventory.config.hidden_size,
            )?,
            n_query_heads: inventory.config.full_attention_heads,
            n_kv_heads: inventory.config.full_attention_kv_heads,
            head_dim: inventory.config.full_attention_head_dim,
            rotary_dim: inventory.config.partial_rotary_dim(),
            rope_theta: super::attention::QWEN35_ROPE_THETA,
            rms_epsilon: inventory.config.rms_norm_eps,
        })
    }

    pub const fn layer_index(&self) -> usize {
        self.layer_index
    }
    pub const fn device_id(&self) -> usize {
        self.device_id
    }
    pub const fn q_layout(&self) -> Qwen35W4Layout {
        self.q_proj.layout
    }
    pub const fn k_layout(&self) -> Qwen35W4Layout {
        self.k_proj.layout
    }
    pub const fn v_layout(&self) -> Qwen35W4Layout {
        self.v_proj.layout
    }
    pub const fn o_layout(&self) -> Qwen35W4Layout {
        self.o_proj.layout
    }
    pub const fn gate_layout(&self) -> Qwen35W4Layout {
        self.gate_proj.layout
    }
    pub const fn up_layout(&self) -> Qwen35W4Layout {
        self.up_proj.layout
    }
    pub const fn down_layout(&self) -> Qwen35W4Layout {
        self.down_proj.layout
    }
    pub fn norm_shapes(&self) -> [[usize; 1]; 4] {
        [
            [self.input_norm.shape().dims()[0]],
            [self.q_norm.shape().dims()[0]],
            [self.k_norm.shape().dims()[0]],
            [self.post_attention_norm.shape().dims()[0]],
        ]
    }

    /// Run the real layer-3 input normalization and packed q/k/v projections.
    /// The result stays on the selected CUDA device and preserves the q/gate
    /// concatenation emitted by the checkpoint's q projection.
    pub fn project_attention_inputs(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
    ) -> Result<Qwen35AttentionProjectionTensors, String> {
        if hidden.device() != Device::Cuda(self.device_id)
            || hidden.dtype() != DType::BF16
            || hidden.shape().dims().len() != 2
            || hidden.shape().dims()[1] != self.q_layout().in_features
        {
            return Err(format!(
                "full-attention hidden must be CUDA BF16 [rows, {}], got {:?} {} on {:?}",
                self.q_layout().in_features,
                hidden.shape().dims(),
                hidden.dtype(),
                hidden.device()
            ));
        }
        let normalized = backend
            .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        Ok(Qwen35AttentionProjectionTensors {
            q_gate: self
                .q_proj
                .project(backend.context(), &normalized)
                .map_err(|error| error.to_string())?,
            k: self
                .k_proj
                .project(backend.context(), &normalized)
                .map_err(|error| error.to_string())?,
            v: self
                .v_proj
                .project(backend.context(), &normalized)
                .map_err(|error| error.to_string())?,
        })
    }

    /// Execute one causal CUDA BF16 token step, including the complete
    /// full-attention and SwiGLU layer order from the reference semantics.
    pub fn decode_token(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
        position: usize,
        state: &mut Qwen35CudaFullAttentionState,
    ) -> Result<Tensor, String> {
        let expected_device = Device::Cuda(self.device_id);
        if backend.device() != expected_device
            || state.device_id != self.device_id
            || hidden.device() != expected_device
            || hidden.dtype() != DType::BF16
            || hidden.shape().dims() != [1, self.input_norm.shape().dims()[0]]
        {
            return Err(format!(
                "full-attention decode requires CUDA{} BF16 [1,{}] hidden, matching state/backend; got {:?} {} on {:?}",
                self.device_id,
                self.input_norm.shape().dims()[0],
                hidden.shape().dims(),
                hidden.dtype(),
                hidden.device()
            ));
        }
        if position != state.seq_len() {
            return Err(format!(
                "full-attention position {position} does not match local KV length {}",
                state.seq_len()
            ));
        }
        if position >= state.max_seq_len {
            return Err(format!(
                "full-attention position {position} exceeds max sequence length {}",
                state.max_seq_len
            ));
        }
        if position > u32::MAX as usize {
            return Err("full-attention position exceeds CUDA RoPE range".into());
        }
        if self.n_query_heads != 24
            || self.n_kv_heads != 4
            || self.head_dim != 256
            || self.rotary_dim != 64
            || self.q_layout().out_features != 2 * self.n_query_heads * self.head_dim
            || self.k_layout().out_features != self.n_kv_heads * self.head_dim
            || self.v_layout().out_features != self.n_kv_heads * self.head_dim
            || self.o_layout().out_features != self.input_norm.shape().dims()[0]
            || self.gate_layout().out_features != self.up_layout().out_features
            || self.down_layout().out_features != self.input_norm.shape().dims()[0]
        {
            return Err(
                "checkpoint full-attention dimensions violate the CUDA layer contract".into(),
            );
        }

        let ctx = backend.context();
        let normalized = backend
            .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let q_gate = self
            .q_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let k = self
            .k_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let v = self
            .v_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;

        let q_width = self.n_query_heads * self.head_dim;
        let q = apxinf_cuda::kernels::elementwise::slice_columns_bf16(ctx, &q_gate, 0, q_width)
            .map_err(|error| error.to_string())?
            .reshape(vec![1, self.n_query_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let gate =
            apxinf_cuda::kernels::elementwise::slice_columns_bf16(ctx, &q_gate, q_width, q_width)
                .map_err(|error| error.to_string())?;
        let k = k
            .reshape(vec![1, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let v = v
            .reshape(vec![1, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;

        let q = q
            .reshape(vec![self.n_query_heads, self.head_dim])
            .and_then(|value| backend.rms_norm(&value, &self.q_norm, self.rms_epsilon))
            .map_err(|error| error.to_string())?
            .reshape(vec![1, self.n_query_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let k = k
            .reshape(vec![self.n_kv_heads, self.head_dim])
            .and_then(|value| backend.rms_norm(&value, &self.k_norm, self.rms_epsilon))
            .map_err(|error| error.to_string())?
            .reshape(vec![1, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let q = apxinf_cuda::kernels::rope::apply_partial_batched(
            ctx,
            &q,
            self.n_query_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            position as u32,
        )
        .map_err(|error| error.to_string())?;
        let k = apxinf_cuda::kernels::rope::apply_partial_batched(
            ctx,
            &k,
            self.n_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            position as u32,
        )
        .map_err(|error| error.to_string())?;

        state
            .cache
            .append(ctx, 0, &k, &v, 1)
            .map_err(|error| error.to_string())?;
        let attended = backend
            .sdpa_decode(
                &q,
                &mut state.cache,
                0,
                self.n_query_heads,
                self.n_kv_heads,
                self.head_dim,
                position + 1,
                state.max_seq_len,
            )
            .map_err(|error| error.to_string())?;
        let gate = apxinf_cuda::kernels::activation::sigmoid(ctx, &gate)
            .map_err(|error| error.to_string())?;
        let gated = backend
            .mul(&attended, &gate)
            .map_err(|error| error.to_string())?;
        let attention_update = self
            .o_proj
            .project(ctx, &gated)
            .map_err(|error| error.to_string())?;
        let residual = backend
            .add(hidden, &attention_update)
            .map_err(|error| error.to_string())?;
        let mlp_input = backend
            .rms_norm(&residual, &self.post_attention_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let mlp_gate = self
            .gate_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_up = self
            .up_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_gate = backend.silu(&mlp_gate).map_err(|error| error.to_string())?;
        let mlp_hidden = backend
            .mul(&mlp_gate, &mlp_up)
            .map_err(|error| error.to_string())?;
        let mlp_update = self
            .down_proj
            .project(ctx, &mlp_hidden)
            .map_err(|error| error.to_string())?;
        let output = backend
            .add(&residual, &mlp_update)
            .map_err(|error| error.to_string())?;
        state.cache.advance(1);
        Ok(output)
    }
}

impl Qwen35CheckpointProjection {
    pub fn from_inventory(
        ctx: &CudaContext,
        inventory: &Qwen35CheckpointInventory,
        base: &str,
    ) -> Result<Self, String> {
        let host = inventory
            .read_packed_linear_payload(base)
            .map_err(|error| format!("load Qwen3.5 projection `{base}`: {error}"))?;
        let layout = Qwen35W4Layout::new(
            host.layout.out_features,
            host.layout.in_features,
            host.layout.group_size,
        )
        .map_err(|error| format!("create Qwen3.5 projection layout: {error}"))?;
        let packed_bytes = host
            .weight_packed
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let scales = Tensor::from_bf16(
            Shape::new(vec![layout.out_features, layout.groups()]),
            &host.scales_bf16,
        )
        .map_err(|error| format!("create Qwen3.5 BF16 scales: {error}"))?;
        let zero_point_bytes = host
            .zero_points
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let projection = Qwen35W4DeviceProjection::upload(
            ctx,
            layout,
            &packed_bytes,
            &scales,
            &zero_point_bytes,
        )
        .map_err(|error| format!("upload Qwen3.5 projection `{base}`: {error}"))?;
        Ok(Self { layout, projection })
    }

    pub const fn layout(&self) -> Qwen35W4Layout {
        self.layout
    }

    pub fn project(&self, ctx: &CudaContext, activation: &Tensor) -> apxinf_core::Result<Tensor> {
        self.projection.project(ctx, activation)
    }

    pub fn project_host_f32(
        &self,
        ctx: &CudaContext,
        activation: &[f32],
    ) -> apxinf_core::Result<Vec<f32>> {
        if activation.len() != self.layout.in_features {
            return Err(apxinf_core::Error::Other(format!(
                "Qwen3.5 projection activation requires {}, got {}",
                self.layout.in_features,
                activation.len()
            )));
        }
        let rounded = activation
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let cpu = Tensor::from_bf16(Shape::new(vec![1, self.layout.in_features]), &rounded)?;
        let gpu = apxinf_cuda::transfers::to_cuda(&cpu, ctx.device_id())?;
        let output = self.project(ctx, &gpu)?;
        let output = apxinf_cuda::transfers::to_cpu(&output)?;
        if output.device() != Device::Cpu || output.dtype() != DType::BF16 {
            return Err(apxinf_core::Error::Other(
                "Qwen3.5 projection returned an invalid output tensor".into(),
            ));
        }
        let bytes = output.storage().as_cpu().ok_or_else(|| {
            apxinf_core::Error::Other("projection output is not CPU storage".into())
        })?;
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_loader::QWEN35_MODEL_REVISION;

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_full_attention_layer_three_owns_all_checkpoint_weights_on_cuda() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();

        let layer = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();

        assert_eq!(layer.layer_index(), 3);
        assert_eq!(layer.device_id(), device);
        assert_eq!(layer.q_layout().in_features, 5120);
        assert_eq!(layer.q_layout().out_features, 12288);
        assert_eq!(layer.k_layout().out_features, 1024);
        assert_eq!(layer.v_layout().out_features, 1024);
        assert_eq!(layer.o_layout().out_features, 5120);
        assert_eq!(layer.gate_layout().out_features, 17408);
        assert_eq!(layer.up_layout().out_features, 17408);
        assert_eq!(layer.down_layout().out_features, 5120);
        assert_eq!(layer.norm_shapes(), [[5120], [256], [256], [5120]]);
        assert_eq!(layer.rope_theta, 10_000_000.0);
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_full_attention_layer_three_projects_cuda_qkv_from_oracle_embedding() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let backend = apxinf_cuda::CudaBackend::new(device).expect("CUDA backend required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let layer = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();
        let oracle = std::path::PathBuf::from(
            "/mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/artifacts/embedding.f32.bin",
        );
        let bytes = std::fs::read(oracle).unwrap();
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        let host = Tensor::from_bf16(
            Shape::new(vec![8, 5120]),
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&host, device).unwrap();
        let projected = layer.project_attention_inputs(&backend, &hidden).unwrap();
        assert_eq!(projected.q_gate.shape().dims(), &[8, 12288]);
        assert_eq!(projected.k.shape().dims(), &[8, 1024]);
        assert_eq!(projected.v.shape().dims(), &[8, 1024]);
        assert_eq!(projected.q_gate.device(), Device::Cuda(device));
        assert!(apxinf_cuda::transfers::to_cpu(&projected.q_gate).is_ok());
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_full_attention_layer_three_runs_one_cuda_token_step_and_advances_local_kv() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let backend = apxinf_cuda::CudaBackend::new(device).expect("CUDA backend required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let layer = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();

        let oracle = std::path::PathBuf::from(
            "/mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/artifacts/embedding.f32.bin",
        );
        let bytes = std::fs::read(oracle).unwrap();
        let values = bytes
            .chunks_exact(4)
            .take(5120)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
        let host = Tensor::from_bf16(Shape::new(vec![1, 5120]), &values).unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&host, device).unwrap();
        let mut state = Qwen35CudaFullAttentionState::new(&backend, 8).unwrap();

        let output = layer
            .decode_token(&backend, &hidden, 0, &mut state)
            .unwrap();

        assert_eq!(output.shape().dims(), &[1, 5120]);
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(output.device(), Device::Cuda(device));
        let output_cpu = apxinf_cuda::transfers::to_cpu(&output).unwrap();
        let output_values = output_cpu.to_f32_vec().unwrap();
        assert!(output_values.iter().all(|value| value.is_finite()));
        assert!(output_values.iter().any(|value| value.abs() > 1e-6));
        assert_eq!(state.cache.storage_dtype(), DType::BF16);
        assert_eq!(state.cache.dtype().unwrap(), Some(DType::BF16));
        assert_eq!(state.seq_len(), 1);
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_layer_zero_projection_matches_cpu_for_selected_outputs() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let base = "model.language_model.layers.0.linear_attn.in_proj_qkv";
        let projection =
            Qwen35CheckpointProjection::from_inventory(&ctx, &inventory, base).unwrap();
        let host = inventory.read_packed_linear(base).unwrap();
        let activation = (0..host.layout.in_features)
            .map(|index| (index as f32 * 0.00031).sin() * 0.02)
            .collect::<Vec<_>>();
        let output = projection.project_host_f32(&ctx, &activation).unwrap();
        assert_eq!(output.len(), host.layout.out_features);
        for out in 0..4 {
            let expected = (0..host.layout.in_features)
                .map(|k| {
                    let value = bf16::from_f32(activation[k]).to_f32();
                    let weight = host
                        .layout
                        .dequantize_value(
                            &host.weight_packed,
                            &host.scales,
                            &host.zero_points,
                            out,
                            k,
                        )
                        .unwrap();
                    value * weight
                })
                .sum::<f32>();
            let expected = bf16::from_f32(expected).to_f32();
            let delta = (output[out] - expected).abs();
            assert!(
                delta <= 0.02 * expected.abs().max(1.0),
                "out={out} delta={delta}"
            );
        }
    }
}
