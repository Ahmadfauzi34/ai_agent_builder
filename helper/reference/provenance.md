# Provenance, receipts, and branches

What this page is for: the engine's evidence machinery — how a run can
be proven, replayed, or branched. Most agents will never need most of
this. Read the "when to ignore it" lines as carefully as the rest.

## Checkpoint (the one most agents need)

- `bin/checkpoint.mjs export` produces a program bundle plus its
  sha256. `verify` checks integrity without running. `restore`
  imports into a fresh registry and runs the spec's inputs.
- Use when: you need to hand a program to someone else, or prove
  that the program you ran later is the program you ran before.
  A bundle whose sha changed is a different program. Treat it as such.
- Do NOT use when: you just need an output value — `run` is enough.

## Execution receipts

- A receipt binds a structural program, signed input claims, observed
  output, and exact serialized state-checkpoint bytes
  (`host-execution-receipt.v2.json`; v1 remains accepted as history).
- Use when: a third party must be able to re-derive that this run
  happened, with these inputs, on this program. Ignore otherwise.

## Signed ingress provenance and replay ledgers

- Ingress provenance binds trusted issuer signatures to an exact
  manifest, subject, port, and f32 input values. A replay ledger is
  optional, host-subject-bound state checked across runner restarts.
- Use when: inputs arrive from an untrusted or multi-party source and
  must be attributable and replay-safe. Ignore when: inputs are your
  own spec's `inputs` field — the runner's own validation and the
  checkpoint bundle already cover that case.

## Checkpoint branches and promotion lineage

- Transactional graph mutation and checkpoint branching let a
  candidate graph be staged, verified against a baseline, and
  promoted only with a passing verification receipt.
- Use when: you mutate a graph and need the promotion to be gated by
  proof, not by assertion. The pack's `verify` tool is the simple
  version of the same idea (baseline vs candidate, ACCEPT/REJECT).
  Ignore the branch machinery unless you are doing staged mutation.

## Rule of thumb

If you cannot state, in one sentence, which third party needs this
evidence and for what decision, you do not need it yet. Start with
`run` + `verify` + `checkpoint`; add provenance machinery only when
a concrete consumer of the evidence appears.
