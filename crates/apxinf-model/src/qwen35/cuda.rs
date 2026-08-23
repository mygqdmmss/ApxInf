use apxinf_core::{Device, DType, Shape, Tensor};
use apxinf_cuda::kernels::qwen35_w4::{Qwen35W4DeviceProjection, Qwen35W4Layout};
use apxinf_cuda::CudaContext;
use half::bf16;

use super::loader::Qwen35CheckpointInventory;

/// Device-owned W4 projection loaded from one checkpoint prefix.
pub struct Qwen35CheckpointProjection {
    layout: Qwen35W4Layout,
    projection: Qwen35W4DeviceProjection,
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
        let bytes = output
            .storage()
            .as_cpu()
            .ok_or_else(|| apxinf_core::Error::Other("projection output is not CPU storage".into()))?;
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
    fn real_layer_zero_projection_matches_cpu_for_selected_outputs() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let inventory = Qwen35CheckpointInventory::from_checkpoint_dir(
            &checkpoint,
            QWEN35_MODEL_REVISION,
        )
        .unwrap();
        let base = "model.language_model.layers.0.linear_attn.in_proj_qkv";
        let projection = Qwen35CheckpointProjection::from_inventory(&ctx, &inventory, base)
            .unwrap();
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
            assert!(delta <= 0.02 * expected.abs().max(1.0), "out={out} delta={delta}");
        }
    }
}
