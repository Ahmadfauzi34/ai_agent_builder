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
    fs::copy(&source, &destination).unwrap_or_else(|error| {
        panic!(
            "failed to copy CFFI header from {} to {}: {error}",
            source.display(),
            destination.display()
        )
    });
}
