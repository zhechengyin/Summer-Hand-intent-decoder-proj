#ifndef FM_CSSD_LDA_H
#define FM_CSSD_LDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FM_CSSD_LDA_CHANNELS 28u
#define FM_CSSD_LDA_SAMPLE_RATE_HZ 100u
#define FM_CSSD_LDA_UPDATE_SAMPLES 5u
#define FM_CSSD_LDA_HISTORY_SAMPLES 40u
#define FM_CSSD_LDA_COLD_START_SAMPLES 50u
#define FM_CSSD_LDA_BP_SOS_SECTIONS 2u
#define FM_CSSD_LDA_ERD_SOS_SECTIONS 4u

typedef enum {
    FM_CSSD_LDA_ERROR = -1,
    FM_CSSD_LDA_WARMING_UP = 0,
    FM_CSSD_LDA_PREDICTION_READY = 1
} fm_cssd_lda_status_t;

typedef struct {
    int32_t class_id;          /* 0 = left, 1 = right. */
    float score;               /* LDA decision score; >= 0 selects right. */
    float probability_right;   /* Logistic transform of score. */
} fm_cssd_lda_output_t;

/*
 * Persistent state for one EEG stream. Allocate it statically or globally on
 * an MCU; do not place it in a small task stack.
 */
typedef struct {
    float bp_filter_state[FM_CSSD_LDA_BP_SOS_SECTIONS]
                         [FM_CSSD_LDA_CHANNELS][2u];
    float erd_filter_state[FM_CSSD_LDA_ERD_SOS_SECTIONS]
                          [FM_CSSD_LDA_CHANNELS][2u];
    float bp_ring[FM_CSSD_LDA_CHANNELS][FM_CSSD_LDA_HISTORY_SAMPLES];
    float erd_ring[FM_CSSD_LDA_CHANNELS][FM_CSSD_LDA_HISTORY_SAMPLES];
    uint16_t samples_seen;
    uint16_t ring_count;
    uint16_t ring_write_index;
    uint8_t initialized;
} fm_cssd_lda_state_t;

/* Frozen input order. The incoming sample[channel] must follow this order. */
extern const char *const fm_cssd_lda_channel_names[FM_CSSD_LDA_CHANNELS];

/* SHA-256 of the NPZ checkpoint used to generate the compiled parameters. */
extern const char fm_cssd_lda_checkpoint_sha256[];

/* Returns the exact number of bytes occupied by one persistent stream state. */
size_t fm_cssd_lda_state_size_bytes(void);

/*
 * Initialize a cold stream using its first 28-channel sample to establish the
 * causal Butterworth steady-state initial condition. This call does not
 * consume the sample: pass the same first sample again to the first push.
 */
fm_cssd_lda_status_t fm_cssd_lda_reset(
    fm_cssd_lda_state_t *state,
    const float first_sample[FM_CSSD_LDA_CHANNELS]
);

/*
 * Consume one sample containing all 28 channels in the frozen channel order.
 * A prediction becomes available after 50 samples. The function can predict
 * on every subsequent sample; production firmware normally reads the result
 * every five samples (50 ms).
 */
fm_cssd_lda_status_t fm_cssd_lda_push_sample(
    fm_cssd_lda_state_t *state,
    const float sample[FM_CSSD_LDA_CHANNELS],
    fm_cssd_lda_output_t *output
);

/*
 * Consume sample-major interleaved data:
 * samples[t * FM_CSSD_LDA_CHANNELS + channel].
 * The returned prediction, when ready, corresponds to the final sample in the
 * block. Use sample_count=5 for the frozen 50 ms update contract.
 */
fm_cssd_lda_status_t fm_cssd_lda_push_block(
    fm_cssd_lda_state_t *state,
    const float *samples,
    size_t sample_count,
    fm_cssd_lda_output_t *output
);

#ifdef __cplusplus
}
#endif

#endif
