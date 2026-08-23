use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GdnError {
    #[error("GDN dimensions are inconsistent")]
    Dimensions,
    #[error("GDN {name} shape is inconsistent")]
    Shape { name: &'static str },
    #[error("GDN {name} contains a non-finite value")]
    NonFinite { name: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDimensions {
    pub conv_kernel: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl GdnDimensions {
    pub fn new(
        conv_kernel: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self, GdnError> {
        if conv_kernel == 0
            || key_heads == 0
            || value_heads == 0
            || key_dim == 0
            || value_dim == 0
            || value_heads % key_heads != 0
        {
            return Err(GdnError::Dimensions);
        }
        Ok(Self {
            conv_kernel,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        })
    }

    pub fn qwen35() -> Result<Self, GdnError> {
        Self::new(4, 16, 48, 128, 128)
    }

    pub const fn conv_channels(self) -> usize {
        self.key_heads * self.key_dim * 2 + self.value_heads * self.value_dim
    }

    const fn recurrent_elements(self) -> usize {
        self.value_heads * self.key_dim * self.value_dim
    }
}

#[derive(Debug, Clone)]
pub struct GdnState {
    dimensions: GdnDimensions,
    conv_ring: Vec<f32>,
    conv_cursor: usize,
    recurrent: Vec<f32>,
}

impl GdnState {
    pub fn new(dimensions: GdnDimensions) -> Result<Self, GdnError> {
        let conv_elements = dimensions
            .conv_channels()
            .checked_mul(dimensions.conv_kernel)
            .ok_or(GdnError::Dimensions)?;
        Ok(Self {
            dimensions,
            conv_ring: vec![0.0; conv_elements],
            conv_cursor: 0,
            recurrent: vec![0.0; dimensions.recurrent_elements()],
        })
    }

    pub fn causal_conv_silu(
        &mut self,
        input: &[f32],
        weights: &[f32],
    ) -> Result<Vec<f32>, GdnError> {
        let channels = self.dimensions.conv_channels();
        if input.len() != channels || weights.len() != channels * self.dimensions.conv_kernel {
            return Err(GdnError::Shape {
                name: "convolution",
            });
        }
        require_finite("convolution input", input)?;
        require_finite("convolution weight", weights)?;
        for (channel, value) in input.iter().enumerate() {
            self.conv_ring[channel * self.dimensions.conv_kernel + self.conv_cursor] = *value;
        }
        self.conv_cursor = (self.conv_cursor + 1) % self.dimensions.conv_kernel;
        let mut output = vec![0.0; channels];
        for channel in 0..channels {
            let base = channel * self.dimensions.conv_kernel;
            let sum = (0..self.dimensions.conv_kernel)
                .map(|offset| {
                    let ring_index = (self.conv_cursor + offset) % self.dimensions.conv_kernel;
                    self.conv_ring[base + ring_index] * weights[base + offset]
                })
                .sum::<f32>();
            output[channel] = sum / (1.0 + (-sum).exp());
        }
        require_finite("convolution output", &output)?;
        Ok(output)
    }

    pub fn conv_ring_channel(&self, channel: usize) -> Result<Vec<f32>, GdnError> {
        if channel >= self.dimensions.conv_channels() {
            return Err(GdnError::Shape {
                name: "convolution channel",
            });
        }
        let base = channel * self.dimensions.conv_kernel;
        Ok((0..self.dimensions.conv_kernel)
            .map(|offset| {
                self.conv_ring[base + (self.conv_cursor + offset) % self.dimensions.conv_kernel]
            })
            .collect())
    }

    pub fn recurrent_step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        g: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, GdnError> {
        let d = self.dimensions;
        if query.len() != d.key_heads * d.key_dim
            || key.len() != d.key_heads * d.key_dim
            || value.len() != d.value_heads * d.value_dim
            || g.len() != d.value_heads
            || beta.len() != d.value_heads
        {
            return Err(GdnError::Shape {
                name: "recurrent step",
            });
        }
        for (name, values) in [
            ("query", query),
            ("key", key),
            ("value", value),
            ("decay", g),
            ("beta", beta),
        ] {
            require_finite(name, values)?;
        }
        let query = l2_normalize_heads(query, d.key_heads, d.key_dim)?;
        let key = l2_normalize_heads(key, d.key_heads, d.key_dim)?;
        let ratio = d.value_heads / d.key_heads;
        let query_scale = (d.key_dim as f32).sqrt().recip();
        let mut next = self.recurrent.clone();
        let mut output = vec![0.0; d.value_heads * d.value_dim];
        for value_head in 0..d.value_heads {
            let key_head = value_head / ratio;
            let q_base = key_head * d.key_dim;
            let state_base = value_head * d.key_dim * d.value_dim;
            let value_base = value_head * d.value_dim;
            let decay = g[value_head].exp();
            for element in &mut next[state_base..state_base + d.key_dim * d.value_dim] {
                *element *= decay;
            }
            for value_dimension in 0..d.value_dim {
                let memory = (0..d.key_dim)
                    .map(|key_dimension| {
                        next[state_base + key_dimension * d.value_dim + value_dimension]
                            * key[q_base + key_dimension]
                    })
                    .sum::<f32>();
                let delta = (value[value_base + value_dimension] - memory) * beta[value_head];
                for key_dimension in 0..d.key_dim {
                    next[state_base + key_dimension * d.value_dim + value_dimension] +=
                        key[q_base + key_dimension] * delta;
                }
                output[value_base + value_dimension] = (0..d.key_dim)
                    .map(|key_dimension| {
                        next[state_base + key_dimension * d.value_dim + value_dimension]
                            * query[q_base + key_dimension]
                            * query_scale
                    })
                    .sum();
            }
        }
        require_finite("recurrent state", &next)?;
        require_finite("recurrent output", &output)?;
        self.recurrent = next;
        Ok(output)
    }

    pub fn eager_prefill(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        g: &[f32],
        beta: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, GdnError> {
        let d = self.dimensions;
        if seq_len == 0
            || query.len() != seq_len * d.key_heads * d.key_dim
            || key.len() != query.len()
            || value.len() != seq_len * d.value_heads * d.value_dim
            || g.len() != seq_len * d.value_heads
            || beta.len() != g.len()
        {
            return Err(GdnError::Shape {
                name: "eager prefill",
            });
        }
        let mut output = Vec::with_capacity(seq_len * d.value_heads * d.value_dim);
        for position in 0..seq_len {
            output.extend(self.recurrent_step(
                &query
                    [position * d.key_heads * d.key_dim..(position + 1) * d.key_heads * d.key_dim],
                &key[position * d.key_heads * d.key_dim..(position + 1) * d.key_heads * d.key_dim],
                &value[position * d.value_heads * d.value_dim
                    ..(position + 1) * d.value_heads * d.value_dim],
                &g[position * d.value_heads..(position + 1) * d.value_heads],
                &beta[position * d.value_heads..(position + 1) * d.value_heads],
            )?);
        }
        Ok(output)
    }

    pub const fn recurrent_dtype(&self) -> &'static str {
        "f32"
    }

    pub fn reset(&mut self) {
        self.conv_ring.fill(0.0);
        self.conv_cursor = 0;
        self.recurrent.fill(0.0);
    }

    pub fn checksum(&self) -> u64 {
        self.conv_ring.iter().chain(&self.recurrent).fold(
            0xcbf29ce484222325u64 ^ self.conv_cursor as u64,
            |hash, value| (hash ^ value.to_bits() as u64).wrapping_mul(0x100000001b3),
        )
    }
}

fn l2_normalize_heads(values: &[f32], heads: usize, head_dim: usize) -> Result<Vec<f32>, GdnError> {
    let mut output = Vec::with_capacity(values.len());
    for head in 0..heads {
        let row = &values[head * head_dim..(head + 1) * head_dim];
        let inverse = (row.iter().map(|value| value * value).sum::<f32>() + 1e-6)
            .sqrt()
            .recip();
        output.extend(row.iter().map(|value| value * inverse));
    }
    require_finite("normalized query/key", &output)?;
    Ok(output)
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), GdnError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(GdnError::NonFinite { name })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_dimensions_match_checkpoint_contract() {
        let dimensions = GdnDimensions::qwen35().unwrap();
        assert_eq!(dimensions.conv_kernel, 4);
        assert_eq!(dimensions.key_heads, 16);
        assert_eq!(dimensions.value_heads, 48);
        assert_eq!(dimensions.key_dim, 128);
        assert_eq!(dimensions.value_dim, 128);
        assert_eq!(dimensions.conv_channels(), 10_240);
    }

    #[test]
    fn convolution_ring_keeps_only_four_causal_positions() {
        let dimensions = GdnDimensions::new(4, 1, 1, 1, 1).unwrap();
        let mut state = GdnState::new(dimensions).unwrap();
        let weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut fifth = Vec::new();
        for value in 1..=5 {
            fifth = state
                .causal_conv_silu(&[value as f32; 3], &weights)
                .unwrap();
        }
        assert_eq!(state.conv_ring_channel(0).unwrap(), [2.0, 3.0, 4.0, 5.0]);
        let expected = 14.0 / (1.0 + (-14.0f32).exp());
        assert!((fifth[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn fp32_recurrent_delta_update_matches_equation() {
        let dimensions = GdnDimensions::new(4, 1, 1, 2, 2).unwrap();
        let mut state = GdnState::new(dimensions).unwrap();
        let output = state
            .recurrent_step(
                &[1.0, 0.0],
                &[1.0, 0.0],
                &[2.0, 4.0],
                &[-2.0f32.ln()],
                &[0.5],
            )
            .unwrap();
        let norm = (1.0 + 1e-6f32).sqrt();
        assert!((output[0] - 1.0 / norm * 1.0 / norm * 1.0 / 2.0f32.sqrt()).abs() < 1e-5);
        assert!((output[1] - 2.0 / norm * 1.0 / norm * 1.0 / 2.0f32.sqrt()).abs() < 1e-5);
        assert_eq!(state.recurrent_dtype(), "f32");
    }

    #[test]
    fn eager_prefill_matches_repeated_single_token_decode() {
        let dimensions = GdnDimensions::new(4, 1, 1, 2, 2).unwrap();
        let query = [1.0, 0.0, 0.0, 1.0];
        let key = [1.0, 0.0, 0.0, 1.0];
        let value = [2.0, 4.0, 6.0, 8.0];
        let g = [-0.5, -0.25];
        let beta = [0.5, 0.25];
        let mut prefill = GdnState::new(dimensions).unwrap();
        let prefill_output = prefill
            .eager_prefill(&query, &key, &value, &g, &beta, 2)
            .unwrap();
        let mut decode = GdnState::new(dimensions).unwrap();
        let mut decode_output = Vec::new();
        for position in 0..2 {
            decode_output.extend(
                decode
                    .recurrent_step(
                        &query[position * 2..position * 2 + 2],
                        &key[position * 2..position * 2 + 2],
                        &value[position * 2..position * 2 + 2],
                        &g[position..position + 1],
                        &beta[position..position + 1],
                    )
                    .unwrap(),
            );
        }
        assert_eq!(prefill_output, decode_output);
        assert_eq!(prefill.checksum(), decode.checksum());
    }

    #[test]
    fn reset_and_failed_step_leave_no_request_state() {
        let dimensions = GdnDimensions::new(4, 1, 1, 2, 2).unwrap();
        let mut state = GdnState::new(dimensions).unwrap();
        let before = state.checksum();
        assert!(state
            .recurrent_step(&[f32::NAN, 0.0], &[1.0, 0.0], &[1.0, 1.0], &[-1.0], &[0.5])
            .is_err());
        assert_eq!(state.checksum(), before);
        state
            .recurrent_step(&[1.0, 0.0], &[1.0, 0.0], &[1.0, 1.0], &[-1.0], &[0.5])
            .unwrap();
        state.reset();
        assert_eq!(state.checksum(), before);
    }
}
