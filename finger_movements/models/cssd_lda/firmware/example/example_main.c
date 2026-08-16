#include "fm_cssd_lda.h"

#include <stddef.h>

/* Keep the roughly 10 KB state out of a small task stack. */
static fm_cssd_lda_state_t decoder;

/* Replace this stub with the board's 28-channel EEG acquisition function. */
static void acquire_eeg_sample(float sample[FM_CSSD_LDA_CHANNELS])
{
    size_t channel;
    for (channel = 0u; channel < FM_CSSD_LDA_CHANNELS; ++channel) {
        sample[channel] = 0.0f;
    }
}

int main(void)
{
    float sample[FM_CSSD_LDA_CHANNELS];
    fm_cssd_lda_output_t output;
    unsigned int samples_in_update = 0u;

    acquire_eeg_sample(sample);
    if (fm_cssd_lda_reset(&decoder, sample) == FM_CSSD_LDA_ERROR) {
        return 1;
    }

    for (;;) {
        fm_cssd_lda_status_t status;

        /* The first acquired sample must also be the first pushed sample. */
        status = fm_cssd_lda_push_sample(&decoder, sample, &output);
        ++samples_in_update;

        if (samples_in_update == FM_CSSD_LDA_UPDATE_SAMPLES) {
            samples_in_update = 0u;
            if (status == FM_CSSD_LDA_PREDICTION_READY) {
                /* Consume output.class_id, output.score, or probability_right. */
            }
        }
        acquire_eeg_sample(sample);
    }
}
