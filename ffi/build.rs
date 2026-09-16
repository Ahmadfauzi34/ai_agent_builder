use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=include/burn_research_ffi.h");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target_dir = out_dir
        .ancestors()
        .nth(4)
        .expect("Cargo OUT_DIR must live below target/<profile>/build/<pkg>/out");

    let source = manifest_dir.join("include").join("burn_research_ffi.h");
    let destination = target_dir.join("header.h");
    let canonical = fs::read_to_string(&source).unwrap_or_else(|error| {
        panic!("failed to read canonical FFI header {}: {error}", source.display())
    });

    // Maturin feeds target/header.h directly to cffi.FFI().cdef(). CFFI does not
    // run a C preprocessor, so the packaging header must contain declarations only.
    // Keep the canonical public C header unchanged and derive this CFFI-only view by
    // removing include guards/includes plus the C++ linkage wrapper.
    let mut cffi_header = String::with_capacity(canonical.len());
    for line in canonical.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed == "extern \"C\" {" || trimmed == "}" {
            continue;
        }
        cffi_header.push_str(line);
        cffi_header.push('\n');
    }

    assert!(
        cffi_header.contains("br_v1_abi_version"),
        "derived CFFI header lost the ABI v1 declarations"
    );
    assert!(
        !cffi_header.lines().any(|line| line.trim_start().starts_with('#')),
        "derived CFFI header must not contain preprocessor directives"
    );
    assert!(
        !cffi_header.contains("extern \"C\""),
        "derived CFFI header must not contain the C++ linkage wrapper"
    );

    fs::write(&destination, cffi_header).unwrap_or_else(|error| {
        panic!(
            "failed to write CFFI declaration header to {}: {error}",
            destination.display()
        )
    });
}
