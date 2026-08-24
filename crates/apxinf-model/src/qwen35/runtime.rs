use super::config::Qwen35ModelConfig;

const PREFILL_CHUNK_TOKENS: usize = 64;

fn prefill_ranges(prompt_tokens: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    (0..prompt_tokens)
        .step_by(PREFILL_CHUNK_TOKENS)
        .map(move |start| start..prompt_tokens.min(start + PREFILL_CHUNK_TOKENS))
}

fn request_capacity(
    prompt_tokens: usize,
    max_new_tokens: usize,
    max_model_len: usize,
) -> Result<usize, String> {
    if prompt_tokens == 0 || max_new_tokens == 0 {
        return Err("Qwen3.5 request requires a non-empty prompt and positive budget".into());
    }
    let capacity = prompt_tokens
        .checked_add(max_new_tokens)
        .ok_or_else(|| "request token budget overflow".to_string())?;
    if capacity > max_model_len {
        return Err(format!(
            "request budget {capacity} exceeds max_model_len {max_model_len}"
        ));
    }
    Ok(capacity)
}

/// Bytes reserved by one request's mutable CUDA state and peak prefill
/// workspace. GDN keeps current, scratch, and backup buffers so a failed
/// speculative launch cannot discard the previous rollback handle. Attention
/// prefill is bounded to one 64-token block, so its score tensor scales with
/// the retained KV length rather than the full prompt squared.
pub fn request_state_bytes(
    config: &Qwen35ModelConfig,
    max_model_len: usize,
) -> Result<usize, String> {
    if max_model_len == 0 {
        return Err("max_model_len must be non-zero".into());
    }
    let gdn_channels = config
        .linear_key_heads
        .checked_mul(config.linear_head_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| {
            value.checked_add(
                config
                    .linear_value_heads
                    .checked_mul(config.linear_head_dim)?,
            )
        })
        .ok_or_else(|| "GDN convolution dimension overflow".to_string())?;
    let conv_bytes = gdn_channels
        .checked_mul(config.linear_conv_kernel_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "GDN convolution byte estimate overflow".to_string())?;
    let recurrent_bytes = config
        .linear_value_heads
        .checked_mul(config.linear_head_dim)
        .and_then(|value| value.checked_mul(config.linear_head_dim))
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "GDN recurrent byte estimate overflow".to_string())?;
    let gdn_per_layer = conv_bytes
        .checked_add(recurrent_bytes)
        .ok_or_else(|| "GDN byte estimate overflow".to_string())?;
    let attention_per_layer = config
        .full_attention_kv_heads
        .checked_mul(max_model_len)
        .and_then(|value| value.checked_mul(config.full_attention_head_dim))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "attention KV byte estimate overflow".to_string())?;
    let chunk = 64usize;
    let qk = chunk
        .checked_mul(config.linear_head_dim)
        .ok_or_else(|| "GDN prefill workspace overflow".to_string())?;
    let values = chunk
        .checked_mul(config.linear_head_dim)
        .ok_or_else(|| "GDN prefill workspace overflow".to_string())?;
    let matrix = chunk
        .checked_mul(chunk)
        .ok_or_else(|| "GDN prefill workspace overflow".to_string())?;
    let workspace_floats = qk
        .checked_mul(2)
        .and_then(|value| value.checked_add(values))
        .and_then(|value| value.checked_add(chunk.checked_mul(2)?))
        .and_then(|value| value.checked_add(matrix))
        .and_then(|value| value.checked_add(values))
        .and_then(|value| value.checked_add(qk))
        .and_then(|value| value.checked_add(values))
        .ok_or_else(|| "GDN prefill workspace overflow".to_string())?;
    let gdn_prefill_workspace = config
        .linear_value_heads
        .checked_mul(workspace_floats)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| "GDN prefill workspace byte estimate overflow".to_string())?;
    let attention_chunk = PREFILL_CHUNK_TOKENS.min(max_model_len);
    let attention_prefill_workspace = config
        .full_attention_heads
        .checked_mul(attention_chunk)
        .and_then(|value| value.checked_mul(max_model_len))
        .and_then(|value| value.checked_mul(2 * 2))
        .ok_or_else(|| "attention prefill workspace byte estimate overflow".to_string())?;
    config
        .gdn_layer_count()
        .checked_mul(gdn_per_layer)
        .and_then(|value| {
            value.checked_add(
                config
                    .full_attention_layer_count()
                    .checked_mul(attention_per_layer)?,
            )
        })
        .and_then(|value| value.checked_add(gdn_prefill_workspace))
        .and_then(|value| value.checked_add(attention_prefill_workspace))
        .ok_or_else(|| "request state byte estimate overflow".to_string())
}

#[cfg(feature = "cuda")]
mod cuda_runtime {
    use std::sync::Arc;

    use apxinf_core::{Backend, Shape, Tensor};
    use apxinf_cuda::CudaBackend;

    use super::super::config::{LayerType, Qwen35ModelConfig};
    use super::super::cuda::{
        upload_standard_norm_payload, Qwen35Bf16Projection, Qwen35CudaFullAttentionLayer,
        Qwen35CudaFullAttentionState, Qwen35CudaGdnLayer, Qwen35CudaGdnState,
    };
    use super::super::loader::Qwen35CheckpointInventory;
    use super::super::model::greedy_argmax;

    enum Layer {
        Gdn(Qwen35CudaGdnLayer),
        Full(Qwen35CudaFullAttentionLayer),
    }

    enum State {
        Gdn(Qwen35CudaGdnState),
        Full(Qwen35CudaFullAttentionState),
    }

    /// Resident CUDA weights for the complete Qwen3.5 text stack.
    pub struct Qwen35CudaModel {
        backend: CudaBackend,
        config: Qwen35ModelConfig,
        max_model_len: usize,
        embedding: Tensor,
        final_norm: Tensor,
        lm_head: Qwen35Bf16Projection,
        layers: Vec<Layer>,
    }

    impl Qwen35CudaModel {
        pub fn from_inventory(
            inventory: &Qwen35CheckpointInventory,
            device_id: usize,
            max_model_len: usize,
        ) -> Result<Arc<Self>, String> {
            if max_model_len == 0 || max_model_len > inventory.config.max_position_embeddings {
                return Err(format!(
                    "max_model_len {max_model_len} is outside checkpoint range [1, {}]",
                    inventory.config.max_position_embeddings
                ));
            }
            let backend = CudaBackend::new(device_id).map_err(|error| error.to_string())?;
            let ctx = backend.context();
            let config = inventory.config.clone();

            let embedding_payload = inventory
                .read_bf16_tensor_payload(
                    "model.language_model.embed_tokens.weight",
                    &[config.vocab_size, config.hidden_size],
                )
                .map_err(|error| error.to_string())?;
            let embedding_host = Tensor::from_bf16(
                Shape::new(embedding_payload.shape.clone()),
                &embedding_payload.values,
            )
            .map_err(|error| format!("create embedding tensor: {error}"))?;
            let embedding = apxinf_cuda::transfers::to_cuda(&embedding_host, device_id)
                .map_err(|error| format!("upload embedding tensor: {error}"))?;

            let final_norm_payload = inventory
                .read_bf16_tensor_payload("model.language_model.norm.weight", &[config.hidden_size])
                .map_err(|error| error.to_string())?;
            let final_norm = upload_standard_norm_payload(
                ctx,
                &final_norm_payload,
                config.hidden_size,
                "final norm",
            )?;
            let lm_payload = inventory
                .read_bf16_tensor_payload(
                    "lm_head.weight",
                    &[config.vocab_size, config.hidden_size],
                )
                .map_err(|error| error.to_string())?;
            let lm_head = Qwen35Bf16Projection::from_payload(ctx, &lm_payload)?;

            let mut layers = Vec::with_capacity(config.num_hidden_layers);
            for layer_index in 0..config.num_hidden_layers {
                let layer = match config.layer_types[layer_index] {
                    LayerType::Gdn => Layer::Gdn(Qwen35CudaGdnLayer::from_inventory(
                        ctx,
                        inventory,
                        layer_index,
                    )?),
                    LayerType::FullAttention => Layer::Full(
                        Qwen35CudaFullAttentionLayer::from_inventory(ctx, inventory, layer_index)?,
                    ),
                };
                layers.push(layer);
            }
            Ok(Arc::new(Self {
                backend,
                config,
                max_model_len,
                embedding,
                final_norm,
                lm_head,
                layers,
            }))
        }

        pub fn device_id(&self) -> usize {
            self.backend.device_id()
        }

        pub fn config(&self) -> &Qwen35ModelConfig {
            &self.config
        }

        pub fn max_model_len(&self) -> usize {
            self.max_model_len
        }

        pub fn memory_info(&self) -> Result<apxinf_cuda::CudaMemoryInfo, String> {
            self.backend.context().memory_info()
        }

        pub fn open(
            self: &Arc<Self>,
            input_ids: &[u32],
            max_new_tokens: usize,
        ) -> Result<Qwen35CudaSession, String> {
            let request_capacity =
                super::request_capacity(input_ids.len(), max_new_tokens, self.max_model_len)?;
            if let Some(token_id) = input_ids
                .iter()
                .copied()
                .find(|token| (*token as usize) >= self.config.vocab_size)
            {
                return Err(format!("token id {token_id} is outside model vocabulary"));
            }
            let states = self.new_states(request_capacity)?;
            let mut session = Qwen35CudaSession {
                model: Arc::clone(self),
                states,
                position: 0,
                emitted: 0,
                max_new_tokens,
                pending: None,
                last_input: None,
                last_hidden: None,
                #[cfg(test)]
                forward_token_calls: 0,
                #[cfg(test)]
                prefill_calls: 0,
            };
            session.prefill(input_ids)?;
            Ok(session)
        }

        fn new_states(&self, request_capacity: usize) -> Result<Vec<State>, String> {
            self.layers
                .iter()
                .map(|layer| match layer {
                    Layer::Gdn(layer) => Ok(State::Gdn(Qwen35CudaGdnState::new(
                        &self.backend,
                        layer.dimensions(),
                    )?)),
                    Layer::Full(_) => Ok(State::Full(Qwen35CudaFullAttentionState::new(
                        &self.backend,
                        request_capacity,
                    )?)),
                })
                .collect()
        }
    }

    pub struct Qwen35CudaSession {
        model: Arc<Qwen35CudaModel>,
        states: Vec<State>,
        position: usize,
        emitted: usize,
        max_new_tokens: usize,
        pending: Option<u32>,
        last_input: Option<u32>,
        last_hidden: Option<Tensor>,
        #[cfg(test)]
        forward_token_calls: usize,
        #[cfg(test)]
        prefill_calls: usize,
    }

    impl Qwen35CudaSession {
        fn prefill(&mut self, input_ids: &[u32]) -> Result<(), String> {
            if input_ids.is_empty() {
                return Err("prefill requires at least one token".into());
            }
            if self.position != 0 {
                return Err("prefill session must start at position zero".into());
            }
            let mut final_row = None;
            for range in super::prefill_ranges(input_ids.len()) {
                let position = range.start;
                let chunk_ids = &input_ids[range.clone()];
                let mut hidden = self
                    .model
                    .backend
                    .embedding(&self.model.embedding, chunk_ids)
                    .map_err(|error| format!("embedding prefill failed: {error}"))?;
                if std::env::var_os("APXINF_DEBUG_HIDDEN_DIR").is_some() {
                    for row in 0..chunk_ids.len() {
                        let row_hidden = row_slice(&self.model.backend, &hidden, row)?;
                        debug_capture_hidden(
                            &self.model.backend,
                            &row_hidden,
                            "embedding",
                            position + row,
                        )?;
                    }
                }
                for (index, layer) in self.model.layers.iter().enumerate() {
                    hidden = match (layer, &mut self.states[index]) {
                        (Layer::Gdn(layer), State::Gdn(state)) => {
                            if state.position() != position {
                                return Err(format!(
                                    "GDN layer {index} position {} does not match prefill block {position}",
                                    state.position()
                                ));
                            }
                            layer
                                .prefill(&self.model.backend, &hidden, state)
                                .map_err(|error| {
                                    format!("GDN layer {index} prefill failed: {error}")
                                })?
                        }
                        (Layer::Full(layer), State::Full(state)) => layer
                            .prefill(&self.model.backend, &hidden, position, state)
                            .map_err(|error| {
                                format!("full-attention layer {index} prefill failed: {error}")
                            })?,
                        _ => return Err(format!("layer {index} and state type disagree")),
                    };
                    if std::env::var_os("APXINF_DEBUG_HIDDEN_DIR").is_some() {
                        for row in 0..chunk_ids.len() {
                            let row_hidden = row_slice(&self.model.backend, &hidden, row)?;
                            debug_capture_hidden(
                                &self.model.backend,
                                &row_hidden,
                                &format!("layer-{index:03}"),
                                position + row,
                            )?;
                        }
                    }
                }
                if range.end == input_ids.len() {
                    final_row = Some(row_slice(
                        &self.model.backend,
                        &hidden,
                        chunk_ids.len() - 1,
                    )?);
                }
                #[cfg(test)]
                {
                    self.prefill_calls += 1;
                }
            }
            let final_row = final_row.ok_or_else(|| "prefill produced no final row".to_string())?;
            let logits = self.logits_from_hidden(&final_row)?;
            self.pending = Some(
                greedy_argmax(&logits, self.model.config.vocab_size)
                    .map_err(|error| error.to_string())?,
            );
            self.position = input_ids.len();
            self.last_input = input_ids.last().copied();
            self.last_hidden = Some(final_row);
            Ok(())
        }

        pub fn next_token(&mut self) -> Result<Option<u32>, String> {
            if self.emitted >= self.max_new_tokens {
                return Ok(None);
            }
            if self.pending.is_none() {
                let token = self
                    .last_input
                    .ok_or_else(|| "decode session has no previous input".to_string())?;
                self.forward_token(token)?;
                let hidden = self
                    .last_hidden
                    .as_ref()
                    .ok_or_else(|| "decode produced no hidden state".to_string())?;
                let logits = self.logits_from_hidden(hidden)?;
                self.pending = Some(
                    greedy_argmax(&logits, self.model.config.vocab_size)
                        .map_err(|error| error.to_string())?,
                );
            }
            let token = self.pending.take().unwrap();
            self.last_input = Some(token);
            self.emitted += 1;
            Ok(Some(token))
        }

        fn forward_token(&mut self, token: u32) -> Result<(), String> {
            #[cfg(test)]
            {
                self.forward_token_calls += 1;
            }
            if self.position >= self.model.max_model_len {
                return Err("decode position exceeds max_model_len".into());
            }
            let mut hidden = self
                .model
                .backend
                .embedding(&self.model.embedding, &[token])
                .map_err(|error| format!("embedding step failed: {error}"))?;
            debug_capture_hidden(&self.model.backend, &hidden, "embedding", self.position)?;
            for (index, layer) in self.model.layers.iter().enumerate() {
                hidden = match (layer, &mut self.states[index]) {
                    (Layer::Gdn(layer), State::Gdn(state)) => layer
                        .decode_token(&self.model.backend, &hidden, state)
                        .map_err(|error| format!("GDN layer {index} failed: {error}"))?,
                    (Layer::Full(layer), State::Full(state)) => layer
                        .decode_token(&self.model.backend, &hidden, self.position, state)
                        .map_err(|error| format!("full-attention layer {index} failed: {error}"))?,
                    _ => return Err(format!("layer {index} and state type disagree")),
                };
                debug_capture_hidden(
                    &self.model.backend,
                    &hidden,
                    &format!("layer-{index:03}"),
                    self.position,
                )?;
            }
            self.position += 1;
            self.last_input = Some(token);
            self.last_hidden = Some(hidden);
            Ok(())
        }

        fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Vec<f32>, String> {
            let normalized = self
                .model
                .backend
                .rms_norm(
                    hidden,
                    &self.model.final_norm,
                    self.model.config.rms_norm_eps,
                )
                .map_err(|error| format!("final RMSNorm failed: {error}"))?;
            let logits = self
                .model
                .lm_head
                .project(self.model.backend.context(), &normalized)
                .map_err(|error| format!("LM head projection failed: {error}"))?;
            let logits = self
                .model
                .backend
                .to_cpu(&logits)
                .map_err(|error| format!("copy logits to host failed: {error}"))?;
            logits
                .to_f32_vec()
                .map_err(|error| format!("decode logits failed: {error}"))
        }
    }

    fn row_slice(backend: &CudaBackend, hidden: &Tensor, row: usize) -> Result<Tensor, String> {
        let dims = hidden.shape().dims();
        if dims.len() != 2 || row >= dims[0] {
            return Err(format!(
                "hidden row slice requires row {row} in matrix, got {:?}",
                dims
            ));
        }
        let flattened = hidden
            .reshape(vec![1, dims[0] * dims[1]])
            .map_err(|error| format!("reshape hidden for row slice failed: {error}"))?;
        apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            backend.context(),
            &flattened,
            row * dims[1],
            dims[1],
        )
        .map_err(|error| format!("copy hidden row {row} failed: {error}"))
    }

    fn debug_capture_hidden(
        backend: &CudaBackend,
        hidden: &Tensor,
        label: &str,
        position: usize,
    ) -> Result<(), String> {
        let Some(directory) = std::env::var_os("APXINF_DEBUG_HIDDEN_DIR") else {
            return Ok(());
        };
        let selected_position = matches!(position, 7 | 35 | 83);
        let selected_layer = matches!(
            label,
            "embedding"
                | "layer-000"
                | "layer-003"
                | "layer-031"
                | "layer-032"
                | "layer-060"
                | "layer-063"
        );
        if !selected_position || !selected_layer {
            return Ok(());
        }
        let cpu = backend
            .to_cpu(hidden)
            .map_err(|error| format!("debug copy {label} position {position}: {error}"))?;
        let values = cpu
            .to_f32_vec()
            .map_err(|error| format!("debug decode {label} position {position}: {error}"))?;
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let path = std::path::Path::new(&directory)
            .join(format!("service-{label}-pos-{position:03}.f32.bin"));
        std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")))
            .map_err(|error| format!("debug create directory: {error}"))?;
        std::fs::write(&path, bytes)
            .map_err(|error| format!("debug write {}: {error}", path.display()))?;
        let (min, max, sum_abs) = values.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY, 0.0f64),
            |(min, max, sum_abs), value| {
                (
                    min.min(*value),
                    max.max(*value),
                    sum_abs + f64::from(value.abs()),
                )
            },
        );
        eprintln!(
            "qwen35 debug hidden {label} position {position}: min={min:.7} max={max:.7} mean_abs={:.7}",
            sum_abs / values.len().max(1) as f64
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use apxinf_loader::QWEN35_MODEL_REVISION;

        #[test]
        #[ignore = "requires GPU and the pinned Qwen3.5 checkpoint"]
        fn open_prefills_prompt_in_bounded_chunks_and_keeps_final_row() {
            let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
                .map(std::path::PathBuf::from)
                .expect("APXINF_QWEN35_CHECKPOINT must point to checkpoint");
            let device = std::env::var("APXINF_CUDA_DEVICE")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            let inventory =
                Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                    .unwrap();
            let model = Qwen35CudaModel::from_inventory(&inventory, device, 80).unwrap();
            let prompt = vec![1; 65];
            let mut session = model.open(&prompt, 2).unwrap();

            assert_eq!(session.prefill_calls, 2);
            assert_eq!(session.forward_token_calls, 0);
            assert_eq!(session.position, prompt.len());
            assert_eq!(session.last_input, Some(*prompt.last().unwrap()));
            assert_eq!(
                session.last_hidden.as_ref().unwrap().shape().dims(),
                &[1, inventory.config.hidden_size]
            );
            for state in &session.states {
                match state {
                    State::Gdn(state) => assert_eq!(state.position(), prompt.len()),
                    State::Full(state) => {
                        assert_eq!(state.seq_len(), prompt.len());
                        assert_eq!(state.max_seq_len(), prompt.len() + 2);
                    }
                }
            }

            assert!(session.next_token().unwrap().is_some());
            assert_eq!(session.forward_token_calls, 0);
            assert_eq!(session.position, prompt.len());
            assert!(session.next_token().unwrap().is_some());
            assert_eq!(session.forward_token_calls, 1);
            assert_eq!(session.position, prompt.len() + 1);
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_runtime::{Qwen35CudaModel, Qwen35CudaSession};

#[cfg(test)]
mod tests {
    use super::super::config::Qwen35ModelConfig;

    #[test]
    fn prefill_plan_bounds_every_block_to_64_tokens() {
        let ranges = super::prefill_ranges(130).collect::<Vec<_>>();
        assert_eq!(ranges, vec![0..64, 64..128, 128..130]);
        assert!(ranges.iter().all(|range| range.len() <= 64));
    }

    #[test]
    fn request_capacity_is_prompt_plus_generation_budget() {
        assert_eq!(super::request_capacity(8_192, 1, 32_768).unwrap(), 8_193);
        assert!(super::request_capacity(32_768, 1, 32_768).is_err());
        assert!(super::request_capacity(1, 0, 32_768).is_err());
    }

    #[test]
    fn request_state_estimate_accounts_for_all_gdn_and_attention_buffers() {
        let raw = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/qwen35-metadata/config.json"),
        )
        .unwrap();
        let config = Qwen35ModelConfig::from_json_str(&raw).unwrap();
        let bytes = super::request_state_bytes(&config, 32_768).unwrap();
        let gdn_channels = config.linear_key_heads * config.linear_head_dim * 2
            + config.linear_value_heads * config.linear_head_dim;
        let gdn_per_layer = 3 * gdn_channels * config.linear_conv_kernel_dim * 2
            + 3 * config.linear_value_heads * config.linear_head_dim * config.linear_head_dim * 4;
        let attention_per_layer =
            2 * config.full_attention_kv_heads * 32_768 * config.full_attention_head_dim * 2;
        let chunk = 64usize;
        let qk = chunk * config.linear_head_dim;
        let values = chunk * config.linear_head_dim;
        let matrix = chunk * chunk;
        let workspace_floats = qk * 2 + values + chunk * 2 + matrix + values + qk + values;
        let attention_prefill_workspace = config.full_attention_heads * chunk * 32_768 * 2 * 2;
        let expected = 48 * gdn_per_layer
            + 16 * attention_per_layer
            + config.linear_value_heads * workspace_floats * 4
            + attention_prefill_workspace;
        assert_eq!(bytes, expected);
    }
}
