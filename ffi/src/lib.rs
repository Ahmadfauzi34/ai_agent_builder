use std::cell::RefCell;
use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::es::optimizer::EsOptimizer;
use burn_research::graph::CompiledGraph;
use burn_research::graph_parameters::GraphParameterBinding;
use burn_research::program_bundle::{export_program_bundle, import_program_bundle};
use burn_research::registry::LayerRegistry;
use burn_research::WasmTensor;

const BR_V1_OK: i32 = 0;
const BR_V1_NULL_POINTER: i32 = 1;
const BR_V1_INVALID_HANDLE_TYPE: i32 = 2;
const BR_V1_INVALID_ARGUMENT: i32 = 3;
const BR_V1_CORE_ERROR: i32 = 4;
const BR_V1_PANIC: i32 = 5;
const BR_V1_BUFFER_TOO_SMALL: i32 = 6;

const CAPABILITIES: &str = concat!(
    "{",
    "\"schema\":\"burn-research.ffi.v1\",",
    "\"abi_version\":1,",
    "\"symbols\":\"br_v1_*\",",
    "\"handles\":\"opaque_owned\",",
    "\"errors\":\"status_code_plus_thread_local_diagnostic\",",
    "\"panic_boundary\":\"catch_unwind\",",
    "\"tensor_transfer\":\"copy_f32_rank4\",",
    "\"parameter_order\":\"GraphParameterBinding\",",
    "\"checkpoint\":\"ProgramBundle\",",
    "\"host_policy\":\"external\"",
    "}"
);

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

#[derive(Debug)]
struct FfiError {
    status: i32,
    message: String,
}

impl FfiError {
    fn new(status: i32, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn null(context: &str) -> Self {
        Self::new(BR_V1_NULL_POINTER, format!("{context}: null pointer"))
    }

    fn invalid_handle(context: &str, expected: &str) -> Self {
        Self::new(
            BR_V1_INVALID_HANDLE_TYPE,
            format!("{context}: expected {expected} handle"),
        )
    }

    fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(BR_V1_INVALID_ARGUMENT, message)
    }

    fn core(message: impl Into<String>) -> Self {
        Self::new(BR_V1_CORE_ERROR, message)
    }

    fn buffer_too_small(context: &str, required: usize, actual: usize) -> Self {
        Self::new(
            BR_V1_BUFFER_TOO_SMALL,
            format!("{context}: destination requires at least {required} elements, got {actual}"),
        )
    }
}

fn set_last_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string Rust panic".to_string()
    }
}

fn ffi_status<F>(operation: F) -> i32
where
    F: FnOnce() -> Result<(), FfiError>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => BR_V1_OK,
        Ok(Err(error)) => {
            set_last_error(error.message);
            error.status
        }
        Err(payload) => {
            set_last_error(format!(
                "burn-research-ffi: intercepted Rust panic: {}",
                panic_text(payload)
            ));
            BR_V1_PANIC
        }
    }
}

enum HandleObject {
    Registry(LayerRegistry),
    LayerSpec(AgentLayerSpec),
    Builder(AgentGraphBuilder),
    Graph(CompiledGraph),
    Binding(GraphParameterBinding),
    Optimizer(EsOptimizer),
    Tensor(WasmTensor),
    F32(Vec<f32>),
    U8(Vec<u8>),
}

#[repr(C)]
pub struct BrV1Handle {
    object: HandleObject,
}

fn boxed(object: HandleObject) -> *mut BrV1Handle {
    Box::into_raw(Box::new(BrV1Handle { object }))
}

unsafe fn put_handle(
    out: *mut *mut BrV1Handle,
    object: HandleObject,
    context: &str,
) -> Result<(), FfiError> {
    if out.is_null() {
        return Err(FfiError::null(context));
    }
    *out = boxed(object);
    Ok(())
}

unsafe fn handle_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a BrV1Handle, FfiError> {
    handle.as_ref().ok_or_else(|| FfiError::null(context))
}

unsafe fn handle_mut<'a>(
    handle: *mut BrV1Handle,
    context: &str,
) -> Result<&'a mut BrV1Handle, FfiError> {
    handle.as_mut().ok_or_else(|| FfiError::null(context))
}

fn ensure_distinct(handles: &[usize], context: &str) -> Result<(), FfiError> {
    for i in 0..handles.len() {
        for j in (i + 1)..handles.len() {
            if handles[i] != 0 && handles[i] == handles[j] {
                return Err(FfiError::invalid_argument(format!(
                    "{context}: aliased handles are not allowed"
                )));
            }
        }
    }
    Ok(())
}

unsafe fn registry_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a LayerRegistry, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::Registry(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "registry")),
    }
}

unsafe fn registry_mut<'a>(
    handle: *mut BrV1Handle,
    context: &str,
) -> Result<&'a mut LayerRegistry, FfiError> {
    match &mut handle_mut(handle, context)?.object {
        HandleObject::Registry(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "registry")),
    }
}

unsafe fn spec_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a AgentLayerSpec, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::LayerSpec(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "layer-spec")),
    }
}

unsafe fn builder_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a AgentGraphBuilder, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::Builder(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "graph-builder")),
    }
}

unsafe fn builder_mut<'a>(
    handle: *mut BrV1Handle,
    context: &str,
) -> Result<&'a mut AgentGraphBuilder, FfiError> {
    match &mut handle_mut(handle, context)?.object {
        HandleObject::Builder(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "graph-builder")),
    }
}

unsafe fn graph_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a CompiledGraph, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::Graph(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "compiled-graph")),
    }
}

unsafe fn binding_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a GraphParameterBinding, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::Binding(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "graph-parameter-binding")),
    }
}

unsafe fn optimizer_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a EsOptimizer, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::Optimizer(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "optimizer")),
    }
}

unsafe fn optimizer_mut<'a>(
    handle: *mut BrV1Handle,
    context: &str,
) -> Result<&'a mut EsOptimizer, FfiError> {
    match &mut handle_mut(handle, context)?.object {
        HandleObject::Optimizer(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "optimizer")),
    }
}

unsafe fn tensor_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a WasmTensor, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::Tensor(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "tensor")),
    }
}

unsafe fn f32_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a Vec<f32>, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::F32(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "f32-buffer")),
    }
}

unsafe fn u8_ref<'a>(
    handle: *const BrV1Handle,
    context: &str,
) -> Result<&'a Vec<u8>, FfiError> {
    match &handle_ref(handle, context)?.object {
        HandleObject::U8(value) => Ok(value),
        _ => Err(FfiError::invalid_handle(context, "u8-buffer")),
    }
}

unsafe fn write_usize(out: *mut usize, value: usize, context: &str) -> Result<(), FfiError> {
    if out.is_null() {
        return Err(FfiError::null(context));
    }
    *out = value;
    Ok(())
}

unsafe fn write_u32(out: *mut u32, value: u32, context: &str) -> Result<(), FfiError> {
    if out.is_null() {
        return Err(FfiError::null(context));
    }
    *out = value;
    Ok(())
}

fn checked_rank4_len(dims: [u32; 4]) -> Result<usize, FfiError> {
    dims.into_iter().try_fold(1usize, |count, dim| {
        count.checked_mul(dim as usize).ok_or_else(|| {
            FfiError::invalid_argument("br_v1_tensor_new_f32: shape element count overflow")
        })
    })
}

unsafe fn input_f32<'a>(
    ptr_value: *const f32,
    len: usize,
    context: &str,
) -> Result<&'a [f32], FfiError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr_value.is_null() {
        return Err(FfiError::null(context));
    }
    Ok(slice::from_raw_parts(ptr_value, len))
}

unsafe fn input_u8<'a>(
    ptr_value: *const u8,
    len: usize,
    context: &str,
) -> Result<&'a [u8], FfiError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr_value.is_null() {
        return Err(FfiError::null(context));
    }
    Ok(slice::from_raw_parts(ptr_value, len))
}

#[no_mangle]
pub extern "C" fn br_v1_abi_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn br_v1_last_error_len() -> usize {
    LAST_ERROR.with(|slot| slot.borrow().len())
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_last_error_copy(dest: *mut c_char, capacity: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let value = slot.borrow();
        let required = value.len();
        if !dest.is_null() && capacity > 0 {
            let copy_len = required.min(capacity.saturating_sub(1));
            ptr::copy_nonoverlapping(value.as_ptr(), dest.cast::<u8>(), copy_len);
            *dest.add(copy_len) = 0;
        }
        required
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_handle_free(handle: *mut BrV1Handle) -> i32 {
    ffi_status(|| {
        if handle.is_null() {
            return Err(FfiError::null("br_v1_handle_free"));
        }
        drop(Box::from_raw(handle));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_capabilities_json(out: *mut *mut BrV1Handle) -> i32 {
    ffi_status(|| put_handle(out, HandleObject::U8(CAPABILITIES.as_bytes().to_vec()), "br_v1_capabilities_json"))
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_registry_new(out: *mut *mut BrV1Handle) -> i32 {
    ffi_status(|| put_handle(out, HandleObject::Registry(LayerRegistry::new()), "br_v1_registry_new"))
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_layer_linear(
    layer_id: u32,
    in_dim: u32,
    out_dim: u32,
    bias: u8,
    out: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        if bias > 1 {
            return Err(FfiError::invalid_argument(
                "br_v1_layer_linear: bias must be 0 or 1",
            ));
        }
        let spec = AgentLayerSpec::linear(layer_id, in_dim, out_dim, bias != 0)
            .map_err(FfiError::core)?;
        put_handle(out, HandleObject::LayerSpec(spec), "br_v1_layer_linear")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_layer_relu(
    layer_id: u32,
    out: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        let spec = AgentLayerSpec::relu(layer_id);
        put_handle(out, HandleObject::LayerSpec(spec), "br_v1_layer_relu")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_registry_init_layer(
    registry: *mut BrV1Handle,
    layer_spec: *const BrV1Handle,
) -> i32 {
    ffi_status(|| {
        ensure_distinct(
            &[registry as usize, layer_spec as usize],
            "br_v1_registry_init_layer",
        )?;
        let spec = spec_ref(layer_spec, "br_v1_registry_init_layer")?;
        registry_mut(registry, "br_v1_registry_init_layer")?
            .init_agent_layer(spec)
            .map_err(FfiError::core)
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_graph_builder_new(
    num_slots: u32,
    out: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        let builder = AgentGraphBuilder::new(num_slots).map_err(FfiError::core)?;
        put_handle(out, HandleObject::Builder(builder), "br_v1_graph_builder_new")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_graph_builder_add_unary(
    builder: *mut BrV1Handle,
    layer_spec: *const BrV1Handle,
    input_slot: u8,
    output_slot: u8,
) -> i32 {
    ffi_status(|| {
        ensure_distinct(
            &[builder as usize, layer_spec as usize],
            "br_v1_graph_builder_add_unary",
        )?;
        let spec = spec_ref(layer_spec, "br_v1_graph_builder_add_unary")?;
        builder_mut(builder, "br_v1_graph_builder_add_unary")?
            .add_unary(spec, input_slot, output_slot)
            .map_err(FfiError::core)
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_graph_builder_set_output(
    builder: *mut BrV1Handle,
    output_slot: u8,
) -> i32 {
    ffi_status(|| {
        builder_mut(builder, "br_v1_graph_builder_set_output")?
            .set_output(output_slot)
            .map_err(FfiError::core)
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_graph_builder_compile(
    builder: *const BrV1Handle,
    registry: *const BrV1Handle,
    out_graph: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        ensure_distinct(
            &[builder as usize, registry as usize],
            "br_v1_graph_builder_compile",
        )?;
        let graph = builder_ref(builder, "br_v1_graph_builder_compile")?
            .compile(registry_ref(registry, "br_v1_graph_builder_compile")?)
            .map_err(FfiError::core)?;
        put_handle(
            out_graph,
            HandleObject::Graph(graph),
            "br_v1_graph_builder_compile",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_graph_program_identity(
    graph: *const BrV1Handle,
    out_utf8: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        let value = graph_ref(graph, "br_v1_graph_program_identity")?
            .program_identity()
            .into_bytes();
        put_handle(out_utf8, HandleObject::U8(value), "br_v1_graph_program_identity")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_tensor_new_f32(
    data: *const f32,
    len: usize,
    d0: u32,
    d1: u32,
    d2: u32,
    d3: u32,
    out_tensor: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        let dims = [d0, d1, d2, d3];
        let expected = checked_rank4_len(dims)?;
        if expected != len {
            return Err(FfiError::invalid_argument(format!(
                "br_v1_tensor_new_f32: shape {:?} requires {expected} floats, got {len}",
                dims
            )));
        }
        let values = input_f32(data, len, "br_v1_tensor_new_f32")?;
        let shape = dims.map(|value| value as usize);
        let tensor = WasmTensor::new(values, &shape);
        put_handle(out_tensor, HandleObject::Tensor(tensor), "br_v1_tensor_new_f32")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_tensor_len(
    tensor: *const BrV1Handle,
    out_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        let len = tensor_ref(tensor, "br_v1_tensor_len")?.to_array().len();
        write_usize(out_len, len, "br_v1_tensor_len")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_tensor_copy_f32(
    tensor: *const BrV1Handle,
    dest: *mut f32,
    dest_len: usize,
) -> i32 {
    ffi_status(|| {
        let values = tensor_ref(tensor, "br_v1_tensor_copy_f32")?.to_array();
        if dest_len < values.len() {
            return Err(FfiError::buffer_too_small(
                "br_v1_tensor_copy_f32",
                values.len(),
                dest_len,
            ));
        }
        if !values.is_empty() && dest.is_null() {
            return Err(FfiError::null("br_v1_tensor_copy_f32"));
        }
        if !values.is_empty() {
            ptr::copy_nonoverlapping(values.as_ptr(), dest, values.len());
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_graph_run(
    graph: *const BrV1Handle,
    registry: *const BrV1Handle,
    input: *const BrV1Handle,
    out_tensor: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        ensure_distinct(
            &[graph as usize, registry as usize, input as usize],
            "br_v1_graph_run",
        )?;
        let output = graph_ref(graph, "br_v1_graph_run")?
            .run(
                registry_ref(registry, "br_v1_graph_run")?,
                tensor_ref(input, "br_v1_graph_run")?,
            )
            .map_err(FfiError::core)?;
        put_handle(out_tensor, HandleObject::Tensor(output), "br_v1_graph_run")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_binding_build(
    graph: *const BrV1Handle,
    registry: *const BrV1Handle,
    out_binding: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        ensure_distinct(
            &[graph as usize, registry as usize],
            "br_v1_binding_build",
        )?;
        let binding = GraphParameterBinding::build(
            graph_ref(graph, "br_v1_binding_build")?,
            registry_ref(registry, "br_v1_binding_build")?,
        )
        .map_err(FfiError::core)?;
        put_handle(
            out_binding,
            HandleObject::Binding(binding),
            "br_v1_binding_build",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_binding_total_len(
    binding: *const BrV1Handle,
    out_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        write_usize(
            out_len,
            binding_ref(binding, "br_v1_binding_total_len")?.total_len(),
            "br_v1_binding_total_len",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_binding_layout_json(
    binding: *const BrV1Handle,
    out_utf8: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        put_handle(
            out_utf8,
            HandleObject::U8(
                binding_ref(binding, "br_v1_binding_layout_json")?
                    .layout_json()
                    .into_bytes(),
            ),
            "br_v1_binding_layout_json",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_binding_identity_json(
    binding: *const BrV1Handle,
    out_utf8: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        put_handle(
            out_utf8,
            HandleObject::U8(
                binding_ref(binding, "br_v1_binding_identity_json")?
                    .identity_json()
                    .into_bytes(),
            ),
            "br_v1_binding_identity_json",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_binding_read_flat(
    binding: *const BrV1Handle,
    graph: *const BrV1Handle,
    registry: *const BrV1Handle,
    out_f32: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        ensure_distinct(
            &[binding as usize, graph as usize, registry as usize],
            "br_v1_binding_read_flat",
        )?;
        let values = binding_ref(binding, "br_v1_binding_read_flat")?
            .read_flat(
                graph_ref(graph, "br_v1_binding_read_flat")?,
                registry_ref(registry, "br_v1_binding_read_flat")?,
            )
            .map_err(FfiError::core)?;
        put_handle(out_f32, HandleObject::F32(values), "br_v1_binding_read_flat")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_binding_apply_flat(
    binding: *const BrV1Handle,
    graph: *const BrV1Handle,
    registry: *mut BrV1Handle,
    candidate: *const f32,
    candidate_len: usize,
) -> i32 {
    ffi_status(|| {
        ensure_distinct(
            &[binding as usize, graph as usize, registry as usize],
            "br_v1_binding_apply_flat",
        )?;
        let values = input_f32(candidate, candidate_len, "br_v1_binding_apply_flat")?;
        let binding_value = binding_ref(binding, "br_v1_binding_apply_flat")?;
        let graph_value = graph_ref(graph, "br_v1_binding_apply_flat")?;
        binding_value
            .apply_flat(
                graph_value,
                registry_mut(registry, "br_v1_binding_apply_flat")?,
                values,
            )
            .map_err(FfiError::core)
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_es_strict(
    dim: u32,
    strategy: u8,
    seed: u32,
    pop: u32,
    sigma: f32,
    has_lr: u8,
    lr: f32,
    out_optimizer: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        if has_lr > 1 {
            return Err(FfiError::invalid_argument(
                "br_v1_es_strict: has_lr must be 0 or 1",
            ));
        }
        let optimizer = EsOptimizer::strict(
            dim,
            strategy,
            seed,
            Some(pop),
            Some(sigma),
            if has_lr == 1 { Some(lr) } else { None },
        )
        .map_err(FfiError::core)?;
        put_handle(
            out_optimizer,
            HandleObject::Optimizer(optimizer),
            "br_v1_es_strict",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_es_ask(
    optimizer: *mut BrV1Handle,
    out_f32: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        let values = optimizer_mut(optimizer, "br_v1_es_ask")?.ask();
        put_handle(out_f32, HandleObject::F32(values), "br_v1_es_ask")
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_es_batch_size(
    optimizer: *const BrV1Handle,
    out_batch_size: *mut u32,
) -> i32 {
    ffi_status(|| {
        write_u32(
            out_batch_size,
            optimizer_ref(optimizer, "br_v1_es_batch_size")?.batch_size(),
            "br_v1_es_batch_size",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_es_tell(
    optimizer: *mut BrV1Handle,
    fitness: *const f32,
    fitness_len: usize,
    out_report_utf8: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        let values = input_f32(fitness, fitness_len, "br_v1_es_tell")?;
        let report = optimizer_mut(optimizer, "br_v1_es_tell")?
            .tell(values)
            .map_err(FfiError::core)?;
        put_handle(
            out_report_utf8,
            HandleObject::U8(report.into_bytes()),
            "br_v1_es_tell",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_es_best(
    optimizer: *const BrV1Handle,
    out_f32: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        put_handle(
            out_f32,
            HandleObject::F32(optimizer_ref(optimizer, "br_v1_es_best")?.best()),
            "br_v1_es_best",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_program_bundle_export(
    graph: *const BrV1Handle,
    registry: *const BrV1Handle,
    include_state: u8,
    out_bytes: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        if include_state > 1 {
            return Err(FfiError::invalid_argument(
                "br_v1_program_bundle_export: include_state must be 0 or 1",
            ));
        }
        ensure_distinct(
            &[graph as usize, registry as usize],
            "br_v1_program_bundle_export",
        )?;
        let bundle = export_program_bundle(
            graph_ref(graph, "br_v1_program_bundle_export")?,
            registry_ref(registry, "br_v1_program_bundle_export")?,
            include_state == 1,
        )
        .map_err(FfiError::core)?;
        put_handle(
            out_bytes,
            HandleObject::U8(bundle),
            "br_v1_program_bundle_export",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_program_bundle_import(
    registry: *mut BrV1Handle,
    bytes: *const u8,
    len: usize,
    out_graph: *mut *mut BrV1Handle,
) -> i32 {
    ffi_status(|| {
        if len == 0 {
            return Err(FfiError::invalid_argument(
                "br_v1_program_bundle_import: bundle must not be empty",
            ));
        }
        let bundle = input_u8(bytes, len, "br_v1_program_bundle_import")?;
        let graph = import_program_bundle(
            registry_mut(registry, "br_v1_program_bundle_import")?,
            bundle,
        )
        .map_err(FfiError::core)?;
        put_handle(
            out_graph,
            HandleObject::Graph(graph),
            "br_v1_program_bundle_import",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_f32_buffer_len(
    buffer: *const BrV1Handle,
    out_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        write_usize(
            out_len,
            f32_ref(buffer, "br_v1_f32_buffer_len")?.len(),
            "br_v1_f32_buffer_len",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_f32_buffer_copy(
    buffer: *const BrV1Handle,
    dest: *mut f32,
    dest_len: usize,
) -> i32 {
    ffi_status(|| {
        let values = f32_ref(buffer, "br_v1_f32_buffer_copy")?;
        if dest_len < values.len() {
            return Err(FfiError::buffer_too_small(
                "br_v1_f32_buffer_copy",
                values.len(),
                dest_len,
            ));
        }
        if !values.is_empty() && dest.is_null() {
            return Err(FfiError::null("br_v1_f32_buffer_copy"));
        }
        if !values.is_empty() {
            ptr::copy_nonoverlapping(values.as_ptr(), dest, values.len());
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_u8_buffer_len(
    buffer: *const BrV1Handle,
    out_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        write_usize(
            out_len,
            u8_ref(buffer, "br_v1_u8_buffer_len")?.len(),
            "br_v1_u8_buffer_len",
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn br_v1_u8_buffer_copy(
    buffer: *const BrV1Handle,
    dest: *mut u8,
    dest_len: usize,
) -> i32 {
    ffi_status(|| {
        let values = u8_ref(buffer, "br_v1_u8_buffer_copy")?;
        if dest_len < values.len() {
            return Err(FfiError::buffer_too_small(
                "br_v1_u8_buffer_copy",
                values.len(),
                dest_len,
            ));
        }
        if !values.is_empty() && dest.is_null() {
            return Err(FfiError::null("br_v1_u8_buffer_copy"));
        }
        if !values.is_empty() {
            ptr::copy_nonoverlapping(values.as_ptr(), dest, values.len());
        }
        Ok(())
    })
}
