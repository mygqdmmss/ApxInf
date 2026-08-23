pub mod admission;
pub mod attention;
pub mod config;
pub mod loader;
pub mod weights;

pub use admission::{
    validate_input_ids, validate_input_ids_with_vocab, validate_total_budget, AdmissionError,
};
pub use attention::{
    apply_output_gate_f32, apply_partial_rope_f32, causal_gqa_attention_f32, causal_gqa_decode_f32,
    embedding_f32, gqa_expand_f32, lm_head_f32, qk_norm_f32, residual_add_f32, rms_norm_f32,
    swiglu_f32, AttentionError, FullAttentionKv, FullAttentionReferenceConfig,
    FullAttentionReferenceLayer, PackedLinearReference,
};
pub use config::{
    LayerType, Qwen35Config, Qwen35ConfigError, Qwen35ModelConfig, IMAGE_TOKEN_ID, MODEL_VOCAB_SIZE,
};
pub use loader::{Qwen35CheckpointInventory, Qwen35LoaderError};
pub use weights::{PackedLinearLayout, WeightLayoutError};
