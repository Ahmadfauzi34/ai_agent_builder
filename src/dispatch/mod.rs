//! Dispatch domain: response dispatch request/requirements/preflight, the response
//! intent execution gate, and the revision dispatch execution adapters.

pub mod agent_response_intent;
pub mod response_dispatch_executor_preflight;
pub mod response_dispatch_request;
pub mod response_dispatch_requirements;
pub mod response_intent_execution_gate;
pub mod revision_dispatch_execution_adapter;
pub mod revision_execution_evidence_rejoin;
