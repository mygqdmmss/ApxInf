//! Fail-closed Qwen3.5 asymmetric packed-W4 reference projection.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::contracts::output_tensor;
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

const REQUIRED_GROUP_SIZE: usize = 32;
const FLAG_NON_FINITE_SCALE: u32 = 1;
const FLAG_NON_FINITE_OUTPUT: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35W4Layout {
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

impl Qwen35W4Layout {
    pub fn new(out_features: usize, in_features: usize, group_size: usize) -> Result<Self> {
        if out_features == 0 || in_features == 0 {
            return Err(Error::Other(
                "Qwen3.5 W4 projection dimensions must be non-zero".into(),
            ));
        }
        if group_size != REQUIRED_GROUP_SIZE {
            return Err(Error::Other(format!(
                "Qwen3.5 W4 group size must be {REQUIRED_GROUP_SIZE}, got {group_size}"
            )));
        }
        Ok(Self {
            out_features,
            in_features,
            group_size,
        })
    }

    pub const fn groups(self) -> usize {
        self.in_features.div_ceil(self.group_size)
    }

    pub const fn packed_k_columns(self) -> usize {
        self.in_features.div_ceil(8)
    }

    pub const fn packed_n_rows(self) -> usize {
        self.out_features.div_ceil(8)
    }

    fn weight_bytes(self) -> Result<usize> {
        checked_product(
            &[self.out_features, self.packed_k_columns(), size_of::<u32>()],
            "packed weight",
        )
    }

    fn zero_point_bytes(self) -> Result<usize> {
        checked_product(
            &[self.packed_n_rows(), self.groups(), size_of::<u32>()],
            "packed zero-point",
        )
    }
}

pub struct Qwen35W4Buffers<'a> {
    pub weight_packed: &'a CudaBuffer,
    pub scales: &'a Tensor,
    pub zero_points: &'a CudaBuffer,
}

/// Owns the device-side buffers for one bounded W4 projection.
///
/// Packed weights and zero-points remain raw I32-compatible bytes because the
/// core tensor dtype set intentionally has no I32 variant. Scales are uploaded
/// as native BF16 and are never expanded to a full-precision model copy.
pub struct Qwen35W4DeviceProjection {
    layout: Qwen35W4Layout,
    weight_packed: CudaBuffer,
    scales: Tensor,
    zero_points: CudaBuffer,
}

impl Qwen35W4DeviceProjection {
    pub fn upload(
        ctx: &CudaContext,
        layout: Qwen35W4Layout,
        weight_packed: &[u8],
        scales_cpu: &Tensor,
        zero_points: &[u8],
    ) -> Result<Self> {
        let expected_weight_bytes = layout.weight_bytes()?;
        let expected_zero_point_bytes = layout.zero_point_bytes()?;
        if weight_packed.len() != expected_weight_bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 W4 packed weight requires {expected_weight_bytes} bytes, got {}",
                weight_packed.len()
            )));
        }
        if zero_points.len() != expected_zero_point_bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 W4 zero-point requires {expected_zero_point_bytes} bytes, got {}",
                zero_points.len()
            )));
        }
        let expected_scale_shape = [layout.out_features, layout.groups()];
        if scales_cpu.device() != Device::Cpu
            || scales_cpu.dtype() != DType::BF16
            || scales_cpu.shape().dims() != expected_scale_shape
        {
            return Err(Error::Other(format!(
                "Qwen3.5 W4 CPU scales must be BF16 {:?}, got {:?} {} on {:?}",
                expected_scale_shape,
                scales_cpu.shape().dims(),
                scales_cpu.dtype(),
                scales_cpu.device()
            )));
        }
        let weight_buffer = CudaBuffer::alloc(expected_weight_bytes, ctx.device_id())
            .map_err(Error::Cuda)?;
        weight_buffer
            .copy_from_host(weight_packed)
            .map_err(Error::Cuda)?;
        let zero_point_buffer = CudaBuffer::alloc(expected_zero_point_bytes, ctx.device_id())
            .map_err(Error::Cuda)?;
        zero_point_buffer
            .copy_from_host(zero_points)
            .map_err(Error::Cuda)?;
        let scales = crate::transfers::to_cuda(scales_cpu, ctx.device_id())?;
        Ok(Self {
            layout,
            weight_packed: weight_buffer,
            scales,
            zero_points: zero_point_buffer,
        })
    }

    pub const fn layout(&self) -> Qwen35W4Layout {
        self.layout
    }

    pub fn project(&self, ctx: &CudaContext, activation: &Tensor) -> Result<Tensor> {
        project_bf16(
            ctx,
            activation,
            Qwen35W4Buffers {
                weight_packed: &self.weight_packed,
                scales: &self.scales,
                zero_points: &self.zero_points,
            },
            self.layout,
        )
    }
}

pub fn project_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    buffers: Qwen35W4Buffers<'_>,
    layout: Qwen35W4Layout,
) -> Result<Tensor> {
    let dims = activation.shape().dims();
    if activation.dtype() != DType::BF16 {
        return Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: activation.dtype(),
        });
    }
    if dims.len() != 2 || dims[0] == 0 || dims[1] != layout.in_features {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 activation shape must be [rows, {}], got {dims:?}",
            layout.in_features
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: activation.device(),
        });
    }
    let expected_scale_shape = [layout.out_features, layout.groups()];
    if buffers.scales.dtype() != DType::BF16 {
        return Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: buffers.scales.dtype(),
        });
    }
    if buffers.scales.shape().dims() != expected_scale_shape {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 scale shape must be {expected_scale_shape:?}, got {:?}",
            buffers.scales.shape().dims()
        )));
    }
    if buffers.scales.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: buffers.scales.device(),
        });
    }
    let rows = dims[0];
    let activation_bytes = checked_product(
        &[rows, layout.in_features, DType::BF16.size_in_bytes()],
        "activation",
    )?;
    let activation_buffer = require_tensor_buffer(ctx, "activation", activation, activation_bytes)?;
    let scale_bytes = checked_product(
        &[
            layout.out_features,
            layout.groups(),
            DType::BF16.size_in_bytes(),
        ],
        "scale",
    )?;
    let scale_buffer = require_tensor_buffer(ctx, "scale", buffers.scales, scale_bytes)?;
    require_exact_buffer(
        ctx,
        "packed weight",
        buffers.weight_packed,
        layout.weight_bytes()?,
    )?;
    require_exact_buffer(
        ctx,
        "N-packed zero-point",
        buffers.zero_points,
        layout.zero_point_bytes()?,
    )?;

    let output_bytes = checked_product(
        &[rows, layout.out_features, DType::BF16.size_in_bytes()],
        "output",
    )?;
    let output = output_buffer(ctx, output_bytes)?;
    let flags = CudaBuffer::alloc_zeros(size_of::<u32>(), ctx.device_id()).map_err(Error::Cuda)?;
    let rows_i32 = checked_i32(rows, "rows")?;
    let out_i32 = checked_i32(layout.out_features, "out_features")?;
    let in_i32 = checked_i32(layout.in_features, "in_features")?;
    let group_i32 = checked_i32(layout.group_size, "group_size")?;
    checked_i32(
        rows.checked_mul(layout.out_features)
            .ok_or_else(|| Error::Other("Qwen3.5 W4 CUDA grid size overflow".into()))?,
        "CUDA grid blocks",
    )?;

    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_w4_project_bf16(
            activation_buffer.ptr(),
            buffers.weight_packed.ptr(),
            scale_buffer.ptr(),
            buffers.zero_points.ptr(),
            output.ptr(),
            flags.ptr(),
            rows_i32,
            out_i32,
            in_i32,
            group_i32,
            ctx.stream().handle(),
        ))
        .map_err(|error| Error::Cuda(format!("Qwen3.5 W4 launch failed: {error}")))?;
    }
    ctx.synchronize()
        .map_err(|error| Error::Cuda(format!("Qwen3.5 W4 synchronize failed: {error}")))?;
    let mut flag_bytes = [0u8; size_of::<u32>()];
    flags
        .copy_to_host(&mut flag_bytes)
        .map_err(|error| Error::Cuda(format!("Qwen3.5 W4 status copy failed: {error}")))?;
    let flag = u32::from_le_bytes(flag_bytes);
    if flag & FLAG_NON_FINITE_SCALE != 0 {
        return Err(Error::Other(
            "Qwen3.5 W4 projection encountered a non-finite W4 scale".into(),
        ));
    }
    if flag & FLAG_NON_FINITE_OUTPUT != 0 {
        return Err(Error::Other(
            "Qwen3.5 W4 projection produced a non-finite W4 output".into(),
        ));
    }
    Ok(output_tensor(
        ctx,
        Shape::new(vec![rows, layout.out_features]),
        DType::BF16,
        output,
    ))
}

fn require_tensor_buffer(
    ctx: &CudaContext,
    name: &str,
    tensor: &Tensor,
    required_bytes: usize,
) -> Result<CudaBuffer> {
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.device() != ctx.device_id() || buffer.len() < required_bytes {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 {name} requires {required_bytes} bytes on CUDA{}, got {} bytes on CUDA{}",
            ctx.device_id(),
            buffer.len(),
            buffer.device()
        )));
    }
    Ok(buffer)
}

fn require_exact_buffer(
    ctx: &CudaContext,
    name: &str,
    buffer: &CudaBuffer,
    expected_bytes: usize,
) -> Result<()> {
    if buffer.device() != ctx.device_id() {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 {name} is on CUDA{}, expected CUDA{}",
            buffer.device(),
            ctx.device_id()
        )));
    }
    if buffer.len() != expected_bytes {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 {name} must contain exactly {expected_bytes} bytes, got {}",
            buffer.len()
        )));
    }
    Ok(())
}

fn checked_product(factors: &[usize], name: &str) -> Result<usize> {
    factors.iter().try_fold(1usize, |value, factor| {
        value
            .checked_mul(*factor)
            .ok_or_else(|| Error::Other(format!("Qwen3.5 W4 {name} size overflow")))
    })
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Other(format!("Qwen3.5 W4 {name} exceeds CUDA i32 range")))
}
