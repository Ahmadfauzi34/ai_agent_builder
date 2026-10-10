//! Differential memory test: native heap vs WASM linear memory.
//!
//! Runs the same logical workload (layer create -> get_state -> load_state ->
//! drop) on the native side with a tracking global allocator, and asserts
//! zero net heap growth across cycles. The WASM side is covered by
//! scripts/audit_wasm_memory.mjs (T3/T4), which asserts stable page counts.
//! If native grows while WASM is stable, the leak is in native-only code;
//! if WASM grows while native is stable, it is in the wasm-bindgen glue.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        DEALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn net_bytes() -> i64 {
    ALLOCATED.load(Ordering::Relaxed) as i64 - DEALLOCATED.load(Ordering::Relaxed) as i64
}

fn reset_counters() {
    ALLOCATED.store(0, Ordering::Relaxed);
    DEALLOCATED.store(0, Ordering::Relaxed);
}

// Re-export the layer type used natively. WasmLinear is the same struct the
// WASM boundary wraps.
use burn_research::layers::linear::WasmLinear;

#[test]
fn native_layer_state_cycles_have_zero_net_heap_growth() {
    // Warmup: stabilize lazy statics, one-time caches.
    for _ in 0..5 {
        let mut layer = WasmLinear::new(8, 4, true);
        let state = layer.get_state().expect("serialize");
        layer.load_state(&state).expect("restore");
    }
    reset_counters();

    for _ in 0..50 {
        let mut layer = WasmLinear::new(8, 4, true);
        let state = layer.get_state().expect("serialize");
        layer.load_state(&state).expect("restore");
        // layer dropped here; state Vec dropped here.
    }

    let net = net_bytes();
    // Allow small slack for allocator-internal bookkeeping; a real leak
    // would grow by ~state_bytes per cycle (hundreds of bytes x 50).
    assert!(
        net.abs() < 4096,
        "native heap grew by {net} bytes over 50 layer state cycles"
    );
}
