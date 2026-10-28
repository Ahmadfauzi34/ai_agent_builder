# Hosts (Node / Python / Rust)

What this page is for: which host languages can drive this engine,
and what each host is actually for. The authoritative source is
`engine-audit/docs/host-support.v1.json`; this page is its summary.

## Node (the host this pack uses)

- Status: supported, verified wasm host adapter, ESM.
- Adapter: `engine-audit/node.mjs` over the wasm-bindgen web target.
- This pack's tools (`bin/*.mjs`) are Node tools. If you are an
  agent receiving this pack, Node is the host you use. No other host
  is needed to validate, run, verify, or checkpoint.
- Interactive runner: `engine-audit/interactive_multi_input_ingress.mjs`
  speaks a JSON-lines op protocol (create, setWeights, bind, run,
  validatePlan, ...). The pack tools already speak it; do not speak
  it yourself unless you are extending the tools.

## Python

- Status: documented host layer with a versioned FFI (C ABI) and a
  typed facade, plus an installed-wheel consumer proof.
- Use when: the consuming system is Python and needs to drive the
  engine as a library. Do NOT use when: you just need to run a spec
  — use the Node tools in this pack instead; adding a Python layer
  for a one-off spec is pure overhead.
- The Python documents live in `engine-audit/docs/` (`python-host-layer.md`,
  `python-ffi-v1.md`, `python-facade-v1.md`, `python-wheel-support.md`).
  They are reference material, not required reading for this pack.

## Rust (native package)

- Status: documented native `.crate` consumer boundary
  (`rust-package-support.md`).
- Use when: the consuming system is Rust and embeds the engine
  natively, without wasm. Do NOT use when: you are an agent using
  this pack — the wasm engine in `engine-audit/` is the reference
  machine here, and the Node tools are its interface.

## What "supported" does NOT mean

Per `host-support.v1.json`: supported means the consumer path was
verified on a recorded matrix. It does not mean the runtime or
toolchain is bundled in this pack, and there is no host ranking —
Node is this pack's host by choice, not because it outranks the others.
