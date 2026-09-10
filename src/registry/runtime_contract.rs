use crate::protocol::{
    PayloadCursor, CONV_CONV1D, CONV_CONV2D, CONV_CONVTRANSPOSE2D, LAYER_CONV,
    LAYER_GHOST, LAYER_POOL, LAYER_SEBLOCK, POOL_ADAPTIVEAVGPOOL2D, POOL_AVGPOOL1D,
    POOL_AVGPOOL2D, POOL_MAXPOOL1D, POOL_MAXPOOL2D,
};

fn require_channels(
    shape: [usize; 4],
    expected: usize,
    context: &str,
) -> Result<(), String> {
    if shape[1] != expected {
        return Err(format!(
            "{context}: expected channel axis 1 size {expected}, got {} for shape {:?}",
            shape[1], shape
        ));
    }
    Ok(())
}

fn require_singleton_width(shape: [usize; 4], context: &str) -> Result<(), String> {
    if shape[3] != 1 {
        return Err(format!(
            "{context}: expected axis 3 to be singleton for 1D operation, got shape {:?}",
            shape
        ));
    }
    Ok(())
}

fn require_kernel_fits(
    input: usize,
    padding: usize,
    kernel: usize,
    axis: usize,
    context: &str,
) -> Result<(), String> {
    if kernel == 0 {
        return Err(format!("{context}: kernel axis {axis} must be > 0"));
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|p| input.checked_add(p))
        .ok_or_else(|| format!("{context}: padded extent overflow on axis {axis}"))?;
    if padded < kernel {
        return Err(format!(
            "{context}: kernel size {kernel} exceeds padded input extent {padded} on axis {axis}"
        ));
    }
    Ok(())
}

fn require_transpose_output_positive(
    input: usize,
    padding: usize,
    kernel: usize,
    stride: usize,
    axis: usize,
    context: &str,
) -> Result<(), String> {
    if input == 0 {
        return Err(format!("{context}: input extent on axis {axis} must be > 0"));
    }
    if kernel == 0 || stride == 0 {
        return Err(format!(
            "{context}: kernel and stride on axis {axis} must be > 0"
        ));
    }
    // Burn wrapper uses dilation=1 and output_padding=0, so:
    // out = (input - 1) * stride - 2 * padding + kernel.
    let output = (input as i128 - 1) * stride as i128
        - 2 * padding as i128
        + kernel as i128;
    if output <= 0 {
        return Err(format!(
            "{context}: transpose convolution would produce non-positive extent {output} on axis {axis}"
        ));
    }
    Ok(())
}

fn validate_conv(
    variant: u8,
    payload: &[u8],
    shape: [usize; 4],
) -> Result<(), String> {
    let mut c = PayloadCursor::new(payload);
    let _id = c.read_u32()?;
    let in_channels = c.read_usize()?;
    let _out_channels = c.read_usize()?;
    let kh = c.read_usize()?;
    let kw = c.read_usize()?;
    let sh = c.read_option_usize()?.unwrap_or(1);
    let sw = c.read_option_usize()?.unwrap_or(1);
    let ph = c.read_option_usize()?.unwrap_or(0);
    let pw = c.read_option_usize()?.unwrap_or(0);

    require_channels(shape, in_channels, "Conv forward")?;
    match variant {
        CONV_CONV1D => {
            require_singleton_width(shape, "Conv1d forward")?;
            require_kernel_fits(shape[2], ph, kh, 2, "Conv1d forward")
        }
        CONV_CONV2D => {
            require_kernel_fits(shape[2], ph, kh, 2, "Conv2d forward")?;
            require_kernel_fits(shape[3], pw, kw, 3, "Conv2d forward")
        }
        CONV_CONVTRANSPOSE2D => {
            require_transpose_output_positive(
                shape[2], ph, kh, sh, 2, "ConvTranspose2d forward",
            )?;
            require_transpose_output_positive(
                shape[3], pw, kw, sw, 3, "ConvTranspose2d forward",
            )
        }
        _ => Err(format!(
            "Conv forward: unknown init variant 0x{variant:02X}"
        )),
    }
}

fn validate_pool(
    variant: u8,
    payload: &[u8],
    shape: [usize; 4],
) -> Result<(), String> {
    let mut c = PayloadCursor::new(payload);
    let _id = c.read_u32()?;
    match variant {
        POOL_MAXPOOL1D | POOL_AVGPOOL1D => {
            let kernel = c.read_usize()?;
            let _stride = c.read_option_usize()?.unwrap_or(kernel.max(1));
            let padding = c.read_option_usize()?.unwrap_or(0);
            require_singleton_width(shape, "Pool1d forward")?;
            require_kernel_fits(shape[2], padding, kernel, 2, "Pool1d forward")
        }
        POOL_MAXPOOL2D | POOL_AVGPOOL2D => {
            let kh = c.read_usize()?;
            let kw = c.read_usize()?;
            let _sh = c.read_option_usize()?.unwrap_or(kh.max(1));
            let _sw = c.read_option_usize()?.unwrap_or(kw.max(1));
            let ph = c.read_option_usize()?.unwrap_or(0);
            let pw = c.read_option_usize()?.unwrap_or(0);
            require_kernel_fits(shape[2], ph, kh, 2, "Pool2d forward")?;
            require_kernel_fits(shape[3], pw, kw, 3, "Pool2d forward")
        }
        POOL_ADAPTIVEAVGPOOL2D => {
            let oh = c.read_usize()?;
            let ow = c.read_usize()?;
            if oh == 0 || ow == 0 {
                return Err(format!(
                    "AdaptiveAvgPool2d forward: output dimensions must be > 0, got [{oh}, {ow}]"
                ));
            }
            if shape[2] == 0 || shape[3] == 0 {
                return Err(format!(
                    "AdaptiveAvgPool2d forward: spatial input dimensions must be > 0, got {:?}",
                    shape
                ));
            }
            Ok(())
        }
        _ => Err(format!(
            "Pool forward: unknown init variant 0x{variant:02X}"
        )),
    }
}

fn validate_ghost(payload: &[u8], shape: [usize; 4]) -> Result<(), String> {
    let mut c = PayloadCursor::new(payload);
    let _id = c.read_u32()?;
    let in_channels = c.read_usize()?;
    let _out_channels = c.read_usize()?;
    let kh = c.read_usize()?;
    let kw = c.read_usize()?;
    let _ratio = c.read_option_usize()?;
    let _sh = c.read_option_usize()?.unwrap_or(1);
    let _sw = c.read_option_usize()?.unwrap_or(1);
    let ph = c.read_option_usize()?.unwrap_or(0);
    let pw = c.read_option_usize()?.unwrap_or(0);

    require_channels(shape, in_channels, "Ghost forward")?;
    require_kernel_fits(shape[2], ph, kh, 2, "Ghost forward")?;
    require_kernel_fits(shape[3], pw, kw, 3, "Ghost forward")
}

fn validate_seblock(payload: &[u8], shape: [usize; 4]) -> Result<(), String> {
    let mut c = PayloadCursor::new(payload);
    let _id = c.read_u32()?;
    let channels = c.read_usize()?;
    let _reduction = c.read_option_usize()?;
    require_channels(shape, channels, "SEBlock forward")?;
    if shape[2] == 0 || shape[3] == 0 {
        return Err(format!(
            "SEBlock forward: spatial input dimensions must be > 0, got {:?}",
            shape
        ));
    }
    Ok(())
}

pub(crate) fn validate_unary_runtime_contract(
    layer_type: u8,
    variant: u8,
    payload: &[u8],
    shape: [usize; 4],
) -> Result<(), String> {
    match layer_type {
        LAYER_CONV => validate_conv(variant, payload, shape),
        LAYER_POOL => validate_pool(variant, payload, shape),
        LAYER_GHOST => validate_ghost(payload, shape),
        LAYER_SEBLOCK => validate_seblock(payload, shape),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{require_kernel_fits, require_transpose_output_positive};

    #[test]
    fn kernel_fit_rejects_kernel_larger_than_padded_input() {
        assert!(require_kernel_fits(1, 0, 3, 2, "test").is_err());
        assert!(require_kernel_fits(1, 1, 3, 2, "test").is_ok());
    }

    #[test]
    fn transpose_extent_rejects_non_positive_output() {
        assert!(require_transpose_output_positive(1, 2, 1, 1, 2, "test").is_err());
        assert!(require_transpose_output_positive(1, 0, 3, 1, 2, "test").is_ok());
    }
}