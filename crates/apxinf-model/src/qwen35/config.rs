pub const MODEL_VOCAB_SIZE: usize = 248_320;
pub const IMAGE_TOKEN_ID: u32 = 248_056;
pub const MODEL_MAX_POSITION: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub image_token_id: u32,
    pub max_position_embeddings: usize,
}

impl Qwen35Config {
    pub const fn frozen() -> Self {
        Self {
            vocab_size: MODEL_VOCAB_SIZE,
            image_token_id: IMAGE_TOKEN_ID,
            max_position_embeddings: MODEL_MAX_POSITION,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frozen_model_constants_match_checkpoint_contract() {
        assert_eq!(super::MODEL_VOCAB_SIZE, 248_320);
        assert_eq!(super::IMAGE_TOKEN_ID, 248_056);
    }
}
