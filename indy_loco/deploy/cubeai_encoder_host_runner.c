/* Host-only validation runner for a generated X-CUBE-AI encoder DLL. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "indy_encoder.h"
#include "indy_encoder_data.h"

static void *aligned_zero(size_t alignment, size_t size) {
  void *value = NULL;
  if (posix_memalign(&value, alignment, size) != 0) return NULL;
  for (size_t index = 0; index < size; ++index) ((uint8_t *)value)[index] = 0;
  return value;
}

static void *read_all(const char *path, size_t expected) {
  FILE *stream = fopen(path, "rb");
  if (!stream) return NULL;
  void *data = aligned_zero(32, expected);
  if (!data || fread(data, 1, expected, stream) != expected) {
    free(data);
    data = NULL;
  }
  fclose(stream);
  return data;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: runner WEIGHTS INPUT OUTPUT SAMPLES\n");
    return 2;
  }
  const size_t samples = (size_t)strtoull(argv[4], NULL, 10);
  void *weights = read_all(argv[1], AI_INDY_ENCODER_DATA_WEIGHTS_SIZE);
  void *activations = aligned_zero(32, AI_INDY_ENCODER_DATA_ACTIVATIONS_SIZE);
  float *input_data = aligned_zero(32, AI_INDY_ENCODER_IN_1_SIZE_BYTES);
  float *output_data = aligned_zero(32, AI_INDY_ENCODER_OUT_1_SIZE_BYTES);
  FILE *input = fopen(argv[2], "rb");
  FILE *output = fopen(argv[3], "wb");
  if (!weights || !activations || !input_data || !output_data || !input || !output) {
    fprintf(stderr, "allocation or file open failed\n");
    return 3;
  }

  ai_handle network = AI_HANDLE_NULL;
  ai_handle activation_handles[] = {AI_HANDLE_PTR(activations)};
  ai_handle weight_handles[] = {AI_HANDLE_PTR(weights)};
  ai_error error = ai_indy_encoder_create_and_init(
      &network, activation_handles, weight_handles);
  if (error.type != AI_ERROR_NONE) {
    fprintf(stderr, "create/init failed: %u/%u\n", error.type, error.code);
    return 4;
  }
  ai_buffer *network_input = ai_indy_encoder_inputs_get(network, NULL);
  ai_buffer *network_output = ai_indy_encoder_outputs_get(network, NULL);
  network_input[0].data = AI_HANDLE_PTR(input_data);
  network_output[0].data = AI_HANDLE_PTR(output_data);
  for (size_t sample = 0; sample < samples; ++sample) {
    if (fread(input_data, 1, AI_INDY_ENCODER_IN_1_SIZE_BYTES, input)
        != AI_INDY_ENCODER_IN_1_SIZE_BYTES) {
      fprintf(stderr, "short input at sample %zu\n", sample);
      return 5;
    }
    if (ai_indy_encoder_run(network, network_input, network_output) != 1) {
      error = ai_indy_encoder_get_error(network);
      fprintf(stderr, "run failed at %zu: %u/%u\n", sample, error.type, error.code);
      return 6;
    }
    if (fwrite(output_data, 1, AI_INDY_ENCODER_OUT_1_SIZE_BYTES, output)
        != AI_INDY_ENCODER_OUT_1_SIZE_BYTES) {
      fprintf(stderr, "short output at sample %zu\n", sample);
      return 7;
    }
  }
  ai_indy_encoder_destroy(network);
  fclose(input);
  fclose(output);
  free(weights);
  free(activations);
  free(input_data);
  free(output_data);
  return 0;
}
