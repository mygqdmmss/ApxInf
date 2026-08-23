use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedLinearLayout {
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WeightLayoutError {
    #[error("group size must be non-zero")]
    ZeroGroupSize,
    #[error("packed tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("weight dtype must be I32 packed storage")]
    InvalidPackedDtype,
    #[error("scale dtype must be BF16")]
    InvalidScaleDtype,
}

impl PackedLinearLayout {
    pub const fn new(out_features: usize, in_features: usize, group_size: usize) -> Self {
        Self {
            out_features,
            in_features,
            group_size,
        }
    }

    pub const fn groups(&self) -> usize {
        if self.group_size == 0 {
            0
        } else {
            self.in_features.div_ceil(self.group_size)
        }
    }

    pub const fn packed_k_columns(&self) -> usize {
        self.in_features.div_ceil(8)
    }

    pub const fn packed_n_rows(&self) -> usize {
        self.out_features.div_ceil(8)
    }

    pub fn validate_shapes(
        &self,
        weight_packed: &[usize],
        weight_scale: &[usize],
        weight_zero_point: &[usize],
    ) -> Result<(), WeightLayoutError> {
        if self.group_size == 0 {
            return Err(WeightLayoutError::ZeroGroupSize);
        }
        let expected_weight = [self.out_features, self.packed_k_columns()];
        let expected_scale = [self.out_features, self.groups()];
        let expected_zero_point = [self.packed_n_rows(), self.groups()];
        for (expected, got) in [
            (expected_weight.as_slice(), weight_packed),
            (expected_scale.as_slice(), weight_scale),
            (expected_zero_point.as_slice(), weight_zero_point),
        ] {
            if expected != got {
                return Err(WeightLayoutError::ShapeMismatch {
                    expected: expected.to_vec(),
                    got: got.to_vec(),
                });
            }
        }
        Ok(())
    }

    pub fn dequantize_value(
        &self,
        weight_packed: &[u32],
        scales: &[f32],
        zero_points: &[u32],
        out: usize,
        k: usize,
    ) -> Option<f32> {
        if out >= self.out_features || k >= self.in_features || self.group_size == 0 {
            return None;
        }
        let packed = *weight_packed.get(out * self.packed_k_columns() + k / 8)?;
        let q = ((packed >> ((k % 8) * 4)) & 0xF) as f32;
        let group = k / self.group_size;
        let scale = *scales.get(out * self.groups() + group)?;
        let packed_zp = *zero_points.get((out / 8) * self.groups() + group)?;
        let zp = ((packed_zp >> ((out % 8) * 4)) & 0xF) as f32;
        Some((q - zp) * scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_k_packed_weight_and_n_packed_zero_point_shapes() {
        let layout = PackedLinearLayout::new(1024, 5120, 32);
        assert_eq!(layout.groups(), 160);
        assert_eq!(layout.packed_k_columns(), 640);
        assert_eq!(layout.packed_n_rows(), 128);
        assert!(layout
            .validate_shapes(&[1024, 640], &[1024, 160], &[128, 160])
            .is_ok());
    }

    #[test]
    fn rejects_zero_point_packed_along_k_instead_of_n() {
        let layout = PackedLinearLayout::new(1024, 5120, 32);
        assert!(matches!(
            layout.validate_shapes(&[1024, 640], &[1024, 160], &[1024, 160]),
            Err(WeightLayoutError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn dequantizes_nibble_with_group_scale_and_n_packed_zero_point() {
        let layout = PackedLinearLayout::new(10, 9, 4);
        let mut packed = vec![0u32; layout.out_features * layout.packed_k_columns()];
        packed[0] = 0x0000_000f;
        packed[9 * layout.packed_k_columns() + 1] = 0;
        let scales = vec![2.0; layout.out_features * layout.groups()];
        let mut zero_points = vec![0u32; layout.packed_n_rows() * layout.groups()];
        zero_points[0] = 0x0000_0003;
        zero_points[layout.groups() + 2] = 0x0000_0030;
        assert_eq!(
            layout.dequantize_value(&packed, &scales, &zero_points, 0, 0),
            Some(24.0)
        );
        assert_eq!(
            layout.dequantize_value(&packed, &scales, &zero_points, 9, 8),
            Some(-6.0)
        );
    }
}
