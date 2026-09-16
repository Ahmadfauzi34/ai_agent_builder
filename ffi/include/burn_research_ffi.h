#ifndef BURN_RESEARCH_FFI_H
#define BURN_RESEARCH_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct br_v1_handle br_v1_handle;
typedef int32_t br_v1_status;

enum {
    BR_V1_OK = 0,
    BR_V1_NULL_POINTER = 1,
    BR_V1_INVALID_HANDLE_TYPE = 2,
    BR_V1_INVALID_ARGUMENT = 3,
    BR_V1_CORE_ERROR = 4,
    BR_V1_PANIC = 5,
    BR_V1_BUFFER_TOO_SMALL = 6
};

uint32_t br_v1_abi_version(void);
size_t br_v1_last_error_len(void);
size_t br_v1_last_error_copy(char *dest, size_t capacity);
br_v1_status br_v1_handle_free(br_v1_handle *handle);
br_v1_status br_v1_capabilities_json(br_v1_handle **out);

br_v1_status br_v1_registry_new(br_v1_handle **out);
br_v1_status br_v1_layer_linear(uint32_t layer_id, uint32_t in_dim, uint32_t out_dim, uint8_t bias, br_v1_handle **out);
br_v1_status br_v1_registry_init_layer(br_v1_handle *registry, const br_v1_handle *layer_spec);

br_v1_status br_v1_graph_builder_new(uint32_t num_slots, br_v1_handle **out);
br_v1_status br_v1_graph_builder_add_unary(br_v1_handle *builder, const br_v1_handle *layer_spec, uint8_t input_slot, uint8_t output_slot);
br_v1_status br_v1_graph_builder_set_output(br_v1_handle *builder, uint8_t output_slot);
br_v1_status br_v1_graph_builder_compile(const br_v1_handle *builder, const br_v1_handle *registry, br_v1_handle **out_graph);
br_v1_status br_v1_graph_program_identity(const br_v1_handle *graph, br_v1_handle **out_utf8);

br_v1_status br_v1_tensor_new_f32(const float *data, size_t len, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, br_v1_handle **out_tensor);
br_v1_status br_v1_tensor_len(const br_v1_handle *tensor, size_t *out_len);
br_v1_status br_v1_tensor_copy_f32(const br_v1_handle *tensor, float *dest, size_t dest_len);
br_v1_status br_v1_graph_run(const br_v1_handle *graph, const br_v1_handle *registry, const br_v1_handle *input, br_v1_handle **out_tensor);

br_v1_status br_v1_binding_build(const br_v1_handle *graph, const br_v1_handle *registry, br_v1_handle **out_binding);
br_v1_status br_v1_binding_total_len(const br_v1_handle *binding, size_t *out_len);
br_v1_status br_v1_binding_layout_json(const br_v1_handle *binding, br_v1_handle **out_utf8);
br_v1_status br_v1_binding_identity_json(const br_v1_handle *binding, br_v1_handle **out_utf8);
br_v1_status br_v1_binding_read_flat(const br_v1_handle *binding, const br_v1_handle *graph, const br_v1_handle *registry, br_v1_handle **out_f32);
br_v1_status br_v1_binding_apply_flat(const br_v1_handle *binding, const br_v1_handle *graph, br_v1_handle *registry, const float *candidate, size_t candidate_len);

br_v1_status br_v1_es_strict(uint32_t dim, uint8_t strategy, uint32_t seed, uint32_t pop, float sigma, uint8_t has_lr, float lr, br_v1_handle **out_optimizer);
br_v1_status br_v1_es_ask(br_v1_handle *optimizer, br_v1_handle **out_f32);
br_v1_status br_v1_es_batch_size(const br_v1_handle *optimizer, uint32_t *out_batch_size);
br_v1_status br_v1_es_tell(br_v1_handle *optimizer, const float *fitness, size_t fitness_len, br_v1_handle **out_report_utf8);
br_v1_status br_v1_es_best(const br_v1_handle *optimizer, br_v1_handle **out_f32);

br_v1_status br_v1_program_bundle_export(const br_v1_handle *graph, const br_v1_handle *registry, uint8_t include_state, br_v1_handle **out_bytes);
br_v1_status br_v1_program_bundle_import(br_v1_handle *registry, const uint8_t *bytes, size_t len, br_v1_handle **out_graph);

br_v1_status br_v1_f32_buffer_len(const br_v1_handle *buffer, size_t *out_len);
br_v1_status br_v1_f32_buffer_copy(const br_v1_handle *buffer, float *dest, size_t dest_len);
br_v1_status br_v1_u8_buffer_len(const br_v1_handle *buffer, size_t *out_len);
br_v1_status br_v1_u8_buffer_copy(const br_v1_handle *buffer, uint8_t *dest, size_t dest_len);

#ifdef __cplusplus
}
#endif

#endif
