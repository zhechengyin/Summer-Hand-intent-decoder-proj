#ifndef B2_DECODER_H
#define B2_DECODER_H
#include <stddef.h>

/* Single-instance, non-reentrant. Each run resets the 50-bin model window. */
#ifdef __cplusplus
extern "C" {
#endif
size_t b2_decoder_activation_bytes(void);
int b2_decoder_init(void *activation_buffer, size_t buffer_bytes);
/* Feature-major normalized input: [192][50]; output is normalized [vx,vy]. */
int b2_decoder_run(const float *normalized_window, float *normalized_velocity);
void b2_decoder_destroy(void);
#ifdef __cplusplus
}
#endif
#endif
