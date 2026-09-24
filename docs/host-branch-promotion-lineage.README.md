# Durable branch promotion lineage

This host-side contract connects a committed durable execution checkpoint to an explicitly selected and verified `CheckpointBranchSet` candidate without moving branch-selection authority into the host.

The safe sequence is deliberately two phase:

1. `BranchPromotionLineage.beginPromotion(...)` validates the committed baseline host receipt, the exact compact WASM branch receipt, and the exact candidate ProgramBundle bytes, then persists a `pending` intent.
2. The caller or agent explicitly invokes `CheckpointBranchSet.commitBranchByReceipt(...)` with its selected branch and authorization flag.
3. The caller exports the resulting live ProgramBundle and calls `completePromotion(...)`. Completion is accepted only when the exact checkpoint digest and program identity match the pending candidate.

A crash after step 1 or step 2 but before step 3 leaves a durable pending intent. The host never infers that a pending intent succeeded.

`BranchPromotionLineage` does not choose a branch, call `commitBranchByReceipt`, sign the branch receipt, or assert equivalence beyond the Burn test vectors represented by the WASM receipt. The sidecar ledger is host-trusted correlation/provenance state, not an authorization service.

The machine-readable contract is `host-branch-promotion-lineage.v1.json`. The CI audit `audit_branch_promotion_lineage.mjs` exercises the sequence with a real packaged WASM multi-input graph, real `CheckpointBranchSet.verifyBranch`, real `commitBranchByReceipt`, and exact exported promoted ProgramBundle bytes.
