use half::bf16;

use apxinf_core::{Shape, Tensor};

use crate::kernels::qwen35_gdn::{
    gated_rms_norm_bf16, require_finite_bf16, Qwen35GdnLayout, Qwen35GdnState,
};
use crate::test_util::{download_bf16_as_fp32, upload_fp32_as_bf16};
use crate::{CudaBuffer, CudaContext};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "index {index}: actual={actual} expected={expected} delta={delta} tolerance={tolerance}"
        );
    }
}

fn tiny_layout() -> Qwen35GdnLayout {
    Qwen35GdnLayout::new(4, 1, 1, 2, 2).unwrap()
}

fn upload_weight(ctx: &CudaContext, values: &[f32], shape: Vec<usize>) -> Tensor {
    let rounded = values
        .iter()
        .copied()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let cpu = Tensor::from_bf16(Shape::new(shape), &rounded).unwrap();
    crate::transfers::to_cuda(&cpu, ctx.device_id()).unwrap()
}

#[test]
fn qwen35_cuda_finite_check_rejects_non_finite_bf16() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let finite = upload_fp32_as_bf16(&ctx, &[1.0, -2.0], vec![1, 2]).unwrap();
    require_finite_bf16(&ctx, &finite, "full-attention prefill output").unwrap();

    let non_finite = upload_fp32_as_bf16(&ctx, &[1.0, f32::NAN], vec![1, 2]).unwrap();
    let error =
        require_finite_bf16(&ctx, &non_finite, "full-attention prefill output").unwrap_err();
    assert!(error
        .to_string()
        .contains("full-attention prefill output contains a non-finite value"));
}

#[test]
fn qwen35_gdn_cuda_causal_depthwise_convolution_tracks_four_token_ring() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let weights = upload_weight(
        &ctx,
        &vec![1.0; layout.conv_channels() * layout.conv_kernel],
        vec![layout.conv_channels(), 1, layout.conv_kernel],
    );

    let mut output = Vec::new();
    for token in 1..=5 {
        let input = upload_fp32_as_bf16(
            &ctx,
            &vec![token as f32; layout.conv_channels()],
            vec![1, layout.conv_channels()],
        )
        .unwrap();
        output = download_bf16_as_fp32(&state.causal_conv_silu(&ctx, &input, &weights).unwrap())
            .unwrap();
    }

    assert_eq!(state.conv_cursor(), 1);
    assert_eq!(state.position(), 5);
    assert_close(
        &state.conv_ring_channel_host(&ctx, 0).unwrap(),
        &[2.0, 3.0, 4.0, 5.0],
        0.0,
    );
    let expected = 14.0 / (1.0 + (-14.0f32).exp());
    assert_close(&output, &vec![expected; layout.conv_channels()], 0.03125);
}

#[test]
fn qwen35_gdn_cuda_convolution_commit_can_be_rolled_back() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let weights = upload_weight(
        &ctx,
        &vec![1.0; layout.conv_channels() * layout.conv_kernel],
        vec![layout.conv_channels(), 1, layout.conv_kernel],
    );
    let input = upload_fp32_as_bf16(
        &ctx,
        &vec![1.0; layout.conv_channels()],
        vec![1, layout.conv_channels()],
    )
    .unwrap();

    state.causal_conv_silu(&ctx, &input, &weights).unwrap();
    state.rollback_last_convolution().unwrap();

    assert_eq!(state.position(), 0);
    assert_eq!(state.conv_cursor(), 0);
    assert_eq!(
        state.conv_ring_channel_host(&ctx, 0).unwrap(),
        vec![0.0; layout.conv_kernel]
    );
}

#[test]
fn qwen35_gdn_cuda_prefill_convolution_matches_transformers_zero_left_padding() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35GdnLayout::new(4, 1, 1, 2, 2).unwrap();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let weights = upload_weight(
        &ctx,
        &vec![1.0; layout.conv_channels() * layout.conv_kernel],
        vec![layout.conv_channels(), 1, layout.conv_kernel],
    );
    let input = upload_fp32_as_bf16(
        &ctx,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // token 0
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, // token 1
            3.0, 6.0, 9.0, 12.0, 15.0, 18.0, // token 2
        ],
        vec![3, layout.conv_channels()],
    )
    .unwrap();

    let output = state
        .causal_conv_silu_prefill(&ctx, &input, &weights)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();
    let expected = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 6.0, 12.0, 18.0, 24.0,
        30.0, 36.0,
    ]
    .iter()
    .map(|value| value * (1.0 + (-value).exp()).recip())
    .collect::<Vec<_>>();
    assert_close(&output, &expected, 0.03125);
    assert_eq!(state.position(), 3);
    assert_eq!(state.conv_cursor(), 3);
    assert_close(
        &state.conv_ring_channel_host(&ctx, 0).unwrap(),
        &[0.0, 1.0, 2.0, 3.0],
        0.0,
    );
}

#[test]
fn qwen35_gdn_cuda_fp32_recurrent_update_matches_reference_equation() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

    let output = state
        .gated_delta_step(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();
    let normalized = (1.0 + 1e-6f32).sqrt().recip();
    let expected_scale = normalized * normalized / 2.0f32.sqrt();
    assert_close(&output, &[1.0 * expected_scale, 2.0 * expected_scale], 0.01);
    assert_eq!(state.recurrent_dtype(), "f32");

    let recurrent = state.recurrent_host(&ctx).unwrap();
    assert_close(&recurrent, &[normalized, 2.0 * normalized, 0.0, 0.0], 1e-5);
}

#[test]
fn qwen35_gdn_cuda_recurrent_matches_transformers_bf16_qk_and_beta_boundaries() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35GdnLayout::new(4, 1, 1, 2, 2).unwrap();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.2969, 0.6992], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[0.8984, -0.4004], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0938, 0.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.1], vec![1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.3], vec![1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[1.2], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[1.0], vec![1]).unwrap();

    let output = state
        .gated_delta_step(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();

    // Transformers 5.15.1's torch_recurrent_gated_delta_rule normalizes q/k
    // in BF16 and materializes sigmoid(b) in BF16 before its FP32 recurrence.
    assert_eq!(
        download_bf16_as_fp32(&output).unwrap(),
        vec![0.5234375, 0.0]
    );
}

#[test]
fn qwen35_gdn_cuda_sequence_recurrent_matches_transformers_chunk_fixture() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.2969, 0.6992, -0.75, 1.125], vec![2, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[0.8984, -0.4004, 1.5, -0.25], vec![2, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0938, 0.0, -1.25, 0.75], vec![2, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.1, -0.2], vec![2, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.3, -0.7], vec![2, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[1.2], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[1.0], vec![1]).unwrap();

    let output = state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();

    // Values are generated from the Transformers torch_chunk_gated_delta_rule
    // reference for this two-row, one-head fixture. The tolerance includes
    // the specified BF16 q/k and beta boundaries, but not a semantic shortcut.
    assert_close(&output, &[0.515625, 0.0, 0.189453125, -0.12011719], 0.01);
    assert_eq!(state.position(), 0);
    assert_eq!(state.recurrent_dtype(), "f32");
    assert_close(
        &state.recurrent_host(&ctx).unwrap(),
        &[-0.39399505, 0.24513245, 0.05948618, -0.04085541],
        0.01,
    );
}

#[test]
fn qwen35_gdn_cuda_sequence_qk_l2norm_matches_transformers_bf16_boundary() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[4.125, 7.59375], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[4.125, 7.59375], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[8.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[-20.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

    let output = state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();

    // Transformers performs the multiply and reduction for l2norm while q/k
    // are BF16. These inputs normalize to [0.474609375, 0.875], not the
    // values obtained by accumulating their squares in FP32.
    assert_close(
        &download_bf16_as_fp32(&output).unwrap(),
        &[0.69921875, 0.0],
        0.001,
    );
    assert_close(
        &state.recurrent_host(&ctx).unwrap(),
        &[0.474609375, 0.0, 0.875, 0.0],
        0.000001,
    );
}

#[test]
fn qwen35_gdn_cuda_sequence_normalization_matches_pure_reference_for_all_rows() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();

    let cases = [
        (32usize, 31usize, 1.421875f32),
        (33, 31, 1.421875),
        (33, 32, 1.421875),
        (96, 95, 1.421875),
        (127, 95, 1.421875),
        (127, 96, 1.421875),
        (128, 127, 1.421875),
    ];

    for (rows, active_row, expected) in cases {
        let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
        let query = [1.2969f32, 0.6992]
            .into_iter()
            .cycle()
            .take(rows * 2)
            .collect::<Vec<_>>();
        let mut value = vec![0.0f32; rows * 2];
        value[active_row * 2] = 2.0;
        let a = vec![-100.0f32; rows];
        let mut b = vec![-100.0f32; rows];
        b[active_row] = 100.0;

        let query = upload_fp32_as_bf16(&ctx, &query, vec![rows, 2]).unwrap();
        let key = upload_fp32_as_bf16(
            &ctx,
            &[1.2969f32, 0.6992]
                .into_iter()
                .cycle()
                .take(rows * 2)
                .collect::<Vec<_>>(),
            vec![rows, 2],
        )
        .unwrap();
        let value = upload_fp32_as_bf16(&ctx, &value, vec![rows, 2]).unwrap();
        let a = upload_fp32_as_bf16(&ctx, &a, vec![rows, 1]).unwrap();
        let b = upload_fp32_as_bf16(&ctx, &b, vec![rows, 1]).unwrap();
        let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
        let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

        let output = state
            .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
            .unwrap();
        let output = download_bf16_as_fp32(&output).unwrap();

        assert_close(
            &output[active_row * 2..active_row * 2 + 2],
            &[expected, 0.0],
            0.001,
        );
    }
}

#[test]
fn qwen35_gdn_cuda_sequence_normalization_is_invariant_to_total_row_count() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();

    fn run_case(ctx: &CudaContext, layout: Qwen35GdnLayout, rows: usize) -> Vec<f32> {
        let active_row = 32usize;
        let mut query = vec![0.0f32; rows * 2];
        let mut key = vec![0.0f32; rows * 2];
        let mut value = vec![0.0f32; rows * 2];
        let a = vec![-100.0f32; rows];
        let mut b = vec![-100.0f32; rows];
        query[active_row * 2..active_row * 2 + 2].copy_from_slice(&[1.2969, 0.6992]);
        key[active_row * 2..active_row * 2 + 2].copy_from_slice(&[1.2969, 0.6992]);
        value[active_row * 2] = 2.0;
        b[active_row] = 100.0;

        let query = upload_fp32_as_bf16(ctx, &query, vec![rows, 2]).unwrap();
        let key = upload_fp32_as_bf16(ctx, &key, vec![rows, 2]).unwrap();
        let value = upload_fp32_as_bf16(ctx, &value, vec![rows, 2]).unwrap();
        let a = upload_fp32_as_bf16(ctx, &a, vec![rows, 1]).unwrap();
        let b = upload_fp32_as_bf16(ctx, &b, vec![rows, 1]).unwrap();
        let a_log = upload_fp32_as_bf16(ctx, &[0.0], vec![1]).unwrap();
        let dt_bias = upload_fp32_as_bf16(ctx, &[0.0], vec![1]).unwrap();
        let mut state = Qwen35GdnState::new(ctx, layout).unwrap();
        let output = state
            .gated_delta_prefill(ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
            .unwrap();
        let output = download_bf16_as_fp32(&output).unwrap();
        output[active_row * 2..active_row * 2 + 2].to_vec()
    }

    let tail = run_case(&ctx, layout, 33);
    let full_block = run_case(&ctx, layout, 64);
    assert_close(&tail, &full_block, 0.001);
}

#[test]
fn qwen35_gdn_cuda_sequence_crosses_transformers_64_token_chunk_boundary() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let rows = 65usize;
    let mut query = Vec::with_capacity(rows * 2);
    let mut key = Vec::with_capacity(rows * 2);
    let mut value = Vec::with_capacity(rows * 2);
    let mut a = Vec::with_capacity(rows);
    let mut b = Vec::with_capacity(rows);
    for token in 0..rows {
        query.extend_from_slice(&[
            ((token % 17) as f32) - 8.0,
            (((3 * token) % 19) as f32) - 9.0,
        ]);
        key.extend_from_slice(&[
            (((5 * token) % 23) as f32) - 11.0,
            (((7 * token) % 29) as f32) - 14.0,
        ]);
        value.extend_from_slice(&[
            (((11 * token) % 31) as f32) - 15.0,
            (((13 * token) % 37) as f32) - 18.0,
        ]);
        a.push(if token % 2 == 0 { -2.0 } else { 2.0 });
        b.push(if token % 3 == 0 { -4.0 } else { 4.0 });
    }
    let query = upload_fp32_as_bf16(&ctx, &query, vec![rows, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &key, vec![rows, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &value, vec![rows, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &a, vec![rows, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &b, vec![rows, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

    let output = state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();

    let expected_rows = [
        (0usize, [-0.189453125, -0.2275390625]),
        (1, [-2.71875, -3.40625]),
        (2, [3.5, 3.90625]),
        (62, [-10.0, 7.21875]),
        (63, [-1.171875, 0.95703125]),
        (64, [4.0, -0.68359375]),
    ];
    for (row, expected) in expected_rows {
        assert_close(&output[row * 2..row * 2 + 2], &expected, 0.01);
    }
    // A direct token recurrence produces about [4.0, -0.68359375] at row
    // 64. This oracle requires the chunk triangular solve and padded second
    // chunk used by torch_chunk_gated_delta_rule.
    assert_close(
        &state.recurrent_host(&ctx).unwrap(),
        &[6.6858883, 0.126413, -2.1652663, 1.2823228],
        0.01,
    );
    assert_eq!(state.position(), 0);
}

#[test]
fn qwen35_gdn_cuda_sequence_non_finite_failure_preserves_recurrent_state() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let initial_state = state.recurrent_host(&ctx).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let committed_state = state.recurrent_host(&ctx).unwrap();
    let committed_position = state.position();

    let bad_query = upload_fp32_as_bf16(&ctx, &[f32::NAN, 0.0], vec![1, 2]).unwrap();
    let error = state
        .gated_delta_prefill(&ctx, &bad_query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap_err();
    assert!(error.to_string().contains("non-finite"));
    assert_eq!(state.position(), committed_position);
    assert_eq!(state.recurrent_host(&ctx).unwrap(), committed_state);

    // A failed speculative prefill must not discard the rollback handle for
    // the last successful recurrent commit.
    state.rollback_last_recurrent().unwrap();
    assert_eq!(state.recurrent_host(&ctx).unwrap(), initial_state);
}

#[test]
fn qwen35_gdn_cuda_recurrent_commit_can_be_rolled_back() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

    let first_value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0], vec![1, 2]).unwrap();
    state
        .gated_delta_step(&ctx, &query, &key, &first_value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let first_state = state.recurrent_host(&ctx).unwrap();

    let second_value = upload_fp32_as_bf16(&ctx, &[6.0, 8.0], vec![1, 2]).unwrap();
    state
        .gated_delta_step(&ctx, &query, &key, &second_value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    assert_ne!(state.recurrent_host(&ctx).unwrap(), first_state);

    state.rollback_last_recurrent().unwrap();
    assert_eq!(state.recurrent_host(&ctx).unwrap(), first_state);
}

#[test]
fn qwen35_gdn_cuda_recurrent_repeats_one_key_head_across_three_value_heads() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35GdnLayout::new(4, 1, 3, 2, 2).unwrap();
    assert_eq!(layout.query_width(), 2);
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], vec![1, 6]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();

    let output = state
        .gated_delta_step(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();
    let scale = (1.0 + 1e-6f32).sqrt().recip().powi(2) / 2.0f32.sqrt();
    assert_close(
        &output,
        &[
            1.0 * scale,
            2.0 * scale,
            3.0 * scale,
            4.0 * scale,
            5.0 * scale,
            6.0 * scale,
        ],
        0.02,
    );
    let recurrent = state.recurrent_host(&ctx).unwrap();
    assert_close(
        &recurrent,
        &[
            (1.0 + 1e-6f32).sqrt().recip(),
            2.0 * (1.0 + 1e-6f32).sqrt().recip(),
            0.0,
            0.0,
            3.0 * (1.0 + 1e-6f32).sqrt().recip(),
            4.0 * (1.0 + 1e-6f32).sqrt().recip(),
            0.0,
            0.0,
            5.0 * (1.0 + 1e-6f32).sqrt().recip(),
            6.0 * (1.0 + 1e-6f32).sqrt().recip(),
            0.0,
            0.0,
        ],
        1e-5,
    );
}

#[test]
fn qwen35_gdn_cuda_gated_rms_norm_uses_direct_weight_and_silu_gate() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input = upload_fp32_as_bf16(&ctx, &[5.25, -4.78125], vec![1, 2]).unwrap();
    let gate = upload_fp32_as_bf16(&ctx, &[6.875, 8.75], vec![1, 2]).unwrap();
    let weight = upload_fp32_as_bf16(&ctx, &[2.21875, 1.4375], vec![2]).unwrap();

    let output = gated_rms_norm_bf16(&ctx, &input, &gate, &weight, 1, 2, 1e-6).unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();

    let rms = ((5.25f32.powi(2) + (-4.78125f32).powi(2)) / 2.0 + 1e-6)
        .sqrt()
        .recip();
    let silu = |value: f32| value / (1.0 + (-value).exp());
    // Transformers' Qwen3.5 RMSNormGated rounds the normalized hidden to
    // BF16 before multiplying the BF16 norm weight, then applies the FP32
    // SiLU gate and rounds the final result to BF16.
    let bf16 = |value: f32| half::bf16::from_f32(value).to_f32();
    let expected = [
        bf16(bf16(5.25 * rms) * bf16(2.21875) * silu(6.875)),
        bf16(bf16(-4.78125 * rms) * bf16(1.4375) * silu(8.75)),
    ];
    assert_close(&output, &expected, 0.0001);
}

#[test]
fn qwen35_gdn_cuda_failed_step_and_reset_do_not_leak_request_state() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    state
        .gated_delta_step(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let before = state.recurrent_host(&ctx).unwrap();
    let before_position = state.position();

    let bad_query = upload_fp32_as_bf16(&ctx, &[f32::NAN, 0.0], vec![1, 2]).unwrap();
    let error = state
        .gated_delta_step(&ctx, &bad_query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap_err();
    assert!(error.to_string().contains("non-finite"));
    assert_eq!(state.position(), before_position);
    assert_eq!(state.recurrent_host(&ctx).unwrap(), before);

    state.reset(&ctx).unwrap();
    assert_eq!(state.position(), 0);
    assert_eq!(state.conv_cursor(), 0);
    assert!(state
        .recurrent_host(&ctx)
        .unwrap()
        .iter()
        .all(|value| *value == 0.0));
    assert!(state
        .conv_ring_channel_host(&ctx, 0)
        .unwrap()
        .iter()
        .all(|value| *value == 0.0));
}

#[test]
fn qwen35_gdn_cuda_rejects_short_state_storage_before_launch() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let short = CudaBuffer::alloc(4, ctx.device_id()).unwrap();
    assert!(Qwen35GdnState::from_buffers_for_test(&ctx, layout, short).is_err());
}
