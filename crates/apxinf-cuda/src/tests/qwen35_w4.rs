use half::bf16;

use apxinf_core::Error;

use crate::kernels::qwen35_w4::{project_bf16, Qwen35W4Buffers, Qwen35W4Layout};
use crate::test_util::{assert_bf16_close_reduction, download_bf16_as_fp32, upload_fp32_as_bf16};
use crate::{CudaBuffer, CudaContext};

fn upload_u32(ctx: &CudaContext, values: &[u32]) -> CudaBuffer {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

fn cpu_reference(
    layout: Qwen35W4Layout,
    activation: &[f32],
    weights: &[u32],
    scales: &[f32],
    zero_points: &[u32],
) -> Vec<f32> {
    let activation: Vec<f32> = activation
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect();
    let scales: Vec<f32> = scales
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect();
    let mut output = vec![0.0; layout.out_features];
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let group = k / layout.group_size;
            let packed = weights[out * layout.packed_k_columns() + k / 8];
            let quantized = ((packed >> (4 * (k % 8))) & 0x0f) as f32;
            let packed_zero = zero_points[(out / 8) * layout.groups() + group];
            let zero_point = ((packed_zero >> (4 * (out % 8))) & 0x0f) as f32;
            output[out] +=
                activation[k] * (quantized - zero_point) * scales[out * layout.groups() + group];
        }
        output[out] = bf16::from_f32(output[out]).to_f32();
    }
    output
}

#[test]
fn qwen35_w4_cuda_projection_matches_k_and_n_packed_cpu_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(9, 35, 32).unwrap();
    let activation: Vec<f32> = (0..layout.in_features)
        .map(|index| index as f32 * 0.03125 - 0.5)
        .collect();
    let mut weights = vec![0u32; layout.out_features * layout.packed_k_columns()];
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let value = ((out * 7 + k * 3) & 0x0f) as u32;
            weights[out * layout.packed_k_columns() + k / 8] |= value << (4 * (k % 8));
        }
    }
    let scales: Vec<f32> = (0..layout.out_features * layout.groups())
        .map(|index| 0.0625 * (1 + index) as f32)
        .collect();
    let mut zero_points = vec![0u32; layout.packed_n_rows() * layout.groups()];
    for out in 0..layout.out_features {
        for group in 0..layout.groups() {
            let value = ((out + 2 * group) & 0x0f) as u32;
            zero_points[(out / 8) * layout.groups() + group] |= value << (4 * (out % 8));
        }
    }
    let expected = cpu_reference(layout, &activation, &weights, &scales, &zero_points);
    let activation = upload_fp32_as_bf16(&ctx, &activation, vec![1, layout.in_features]).unwrap();
    let scales =
        upload_fp32_as_bf16(&ctx, &scales, vec![layout.out_features, layout.groups()]).unwrap();
    let weights = upload_u32(&ctx, &weights);
    let zero_points = upload_u32(&ctx, &zero_points);

    let output = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[test]
fn qwen35_w4_cuda_projection_rejects_non_finite_scale() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 1, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[f32::INFINITY], vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &[1]);
    let zero_points = upload_u32(&ctx, &[0]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Other(message) if message.contains("non-finite W4 scale")));
}

#[test]
fn qwen35_w4_cuda_projection_rejects_non_finite_output() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 2, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[f32::MAX, f32::MAX], vec![1, 2]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &[0x0000_00ff]);
    let zero_points = upload_u32(&ctx, &[0]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Other(message) if message.contains("non-finite W4 output")));
}

#[test]
fn qwen35_w4_cuda_projection_validates_shape_dtype_and_storage() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(2, 9, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[1.0; 8], vec![1, 8]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[1.0; 2], vec![2, 1]).unwrap();
    let weights = upload_u32(&ctx, &[0; 4]);
    let zero_points = upload_u32(&ctx, &[0; 1]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Other(message) if message.contains("activation shape")));
}
