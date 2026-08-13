#include "fm_cssd_lda.h"
#include "fm_cssd_lda_params.h"

#include <math.h>
#include <string.h>

#define FM_BP_FEATURES 8u
#define FM_ERD_FEATURES 8u
#define FM_TREND_FEATURES 38u
#define FM_BRANCHES 3u
#define FM_ERD_RECENT_SAMPLES 32u
#define FM_ERD_POOL_SAMPLES 8u
#define FM_TREND_CHANNELS 19u

static int fm_is_finite_sample(const float sample[FM_CSSD_LDA_CHANNELS])
{
    size_t channel;
    for (channel = 0u; channel < FM_CSSD_LDA_CHANNELS; ++channel) {
        if (!isfinite(sample[channel])) {
            return 0;
        }
    }
    return 1;
}

static float fm_filter_one(
    const float sos[][6u],
    size_t sections,
    float state[][FM_CSSD_LDA_CHANNELS][2u],
    size_t channel,
    float input)
{
    size_t section;
    float value = input;

    for (section = 0u; section < sections; ++section) {
        const float output = sos[section][0u] * value + state[section][channel][0u];
        const float next_state_0 = sos[section][1u] * value
                                 - sos[section][4u] * output
                                 + state[section][channel][1u];
        const float next_state_1 = sos[section][2u] * value
                                 - sos[section][5u] * output;
        state[section][channel][0u] = next_state_0;
        state[section][channel][1u] = next_state_1;
        value = output;
    }
    return value;
}

static size_t fm_ring_index(const fm_cssd_lda_state_t *state, size_t offset)
{
    return (state->ring_write_index + offset) % FM_CSSD_LDA_HISTORY_SAMPLES;
}

static float fm_linear_score(
    const float *features,
    const float *weights,
    size_t feature_count,
    float bias)
{
    size_t index;
    float score = bias;
    for (index = 0u; index < feature_count; ++index) {
        score += features[index] * weights[index];
    }
    return score;
}

static float fm_sigmoid(float score)
{
    if (score >= 0.0f) {
        return 1.0f / (1.0f + expf(-score));
    }
    {
        const float exponent = expf(score);
        return exponent / (1.0f + exponent);
    }
}

static fm_cssd_lda_status_t fm_predict(
    const fm_cssd_lda_state_t *state,
    fm_cssd_lda_output_t *output)
{
    float bp_features[FM_BP_FEATURES];
    float erd_features[FM_ERD_FEATURES];
    float trend_features[FM_TREND_FEATURES];
    float branch_scores[FM_BRANCHES];
    size_t pattern;
    size_t time;
    size_t channel;

    if (state == NULL || output == NULL || state->initialized == 0u) {
        return FM_CSSD_LDA_ERROR;
    }
    if (state->samples_seen < FM_CSSD_LDA_COLD_START_SAMPLES
        || state->ring_count < FM_CSSD_LDA_HISTORY_SAMPLES) {
        return FM_CSSD_LDA_WARMING_UP;
    }

    for (pattern = 0u; pattern < 2u; ++pattern) {
        for (time = 0u; time < 4u; ++time) {
            float projected = 0.0f;
            const size_t ring_time = fm_ring_index(state, 36u + time);
            for (channel = 0u; channel < FM_CSSD_LDA_CHANNELS; ++channel) {
                const size_t oldest = fm_ring_index(state, 0u);
                const float rereferenced = state->bp_ring[channel][ring_time]
                                          - state->bp_ring[channel][oldest];
                projected += fm_bp_spatial_filters[pattern][channel] * rereferenced;
            }
            bp_features[pattern * 4u + time] = projected;
        }
    }

    for (pattern = 0u; pattern < 2u; ++pattern) {
        size_t pool;
        for (pool = 0u; pool < 4u; ++pool) {
            float magnitude_sum = 0.0f;
            for (time = 0u; time < FM_ERD_POOL_SAMPLES; ++time) {
                float projected = 0.0f;
                const size_t offset = (FM_CSSD_LDA_HISTORY_SAMPLES
                                     - FM_ERD_RECENT_SAMPLES)
                                    + pool * FM_ERD_POOL_SAMPLES + time;
                const size_t ring_time = fm_ring_index(state, offset);
                for (channel = 0u; channel < FM_CSSD_LDA_CHANNELS; ++channel) {
                    projected += fm_erd_spatial_filters[pattern][channel]
                               * state->erd_ring[channel][ring_time];
                }
                magnitude_sum += fabsf(projected);
            }
            erd_features[pattern * 4u + pool] =
                magnitude_sum / (float)FM_ERD_POOL_SAMPLES;
        }
    }

    for (channel = 0u; channel < FM_TREND_CHANNELS; ++channel) {
        const size_t source_channel = fm_trend_indices[channel];
        const size_t oldest_index = fm_ring_index(state, 0u);
        const float baseline = state->bp_ring[source_channel][oldest_index];
        float oldest_sum = 0.0f;
        float recent_sum = 0.0f;

        for (time = 0u; time < 8u; ++time) {
            oldest_sum += state->bp_ring[source_channel][fm_ring_index(state, time)]
                        - baseline;
        }
        for (time = 30u; time < FM_CSSD_LDA_HISTORY_SAMPLES; ++time) {
            recent_sum += state->bp_ring[source_channel][fm_ring_index(state, time)]
                        - baseline;
        }
        trend_features[channel * 2u] = oldest_sum / 8.0f;
        trend_features[channel * 2u + 1u] = recent_sum / 10.0f;
    }

    branch_scores[0u] = fm_linear_score(
        bp_features, fm_bp_lda_weights, FM_BP_FEATURES, fm_bp_lda_bias);
    branch_scores[1u] = fm_linear_score(
        erd_features, fm_erd_lda_weights, FM_ERD_FEATURES, fm_erd_lda_bias);
    branch_scores[2u] = fm_linear_score(
        trend_features, fm_trend_lda_weights, FM_TREND_FEATURES,
        fm_trend_lda_bias);

    output->score = fm_linear_score(
        branch_scores, fm_fusion_weights, FM_BRANCHES, fm_fusion_bias);
    output->class_id = (output->score >= 0.0f) ? 1 : 0;
    output->probability_right = fm_sigmoid(output->score);
    return FM_CSSD_LDA_PREDICTION_READY;
}

size_t fm_cssd_lda_state_size_bytes(void)
{
    return sizeof(fm_cssd_lda_state_t);
}

fm_cssd_lda_status_t fm_cssd_lda_reset(
    fm_cssd_lda_state_t *state,
    const float first_sample[FM_CSSD_LDA_CHANNELS])
{
    size_t section;
    size_t channel;

    if (state == NULL || first_sample == NULL || !fm_is_finite_sample(first_sample)) {
        return FM_CSSD_LDA_ERROR;
    }
    memset(state, 0, sizeof(*state));

    for (section = 0u; section < FM_CSSD_LDA_BP_SOS_SECTIONS; ++section) {
        for (channel = 0u; channel < FM_CSSD_LDA_CHANNELS; ++channel) {
            state->bp_filter_state[section][channel][0u] =
                fm_bp_initial_state[section][0u] * first_sample[channel];
            state->bp_filter_state[section][channel][1u] =
                fm_bp_initial_state[section][1u] * first_sample[channel];
        }
    }
    for (section = 0u; section < FM_CSSD_LDA_ERD_SOS_SECTIONS; ++section) {
        for (channel = 0u; channel < FM_CSSD_LDA_CHANNELS; ++channel) {
            state->erd_filter_state[section][channel][0u] =
                fm_erd_initial_state[section][0u] * first_sample[channel];
            state->erd_filter_state[section][channel][1u] =
                fm_erd_initial_state[section][1u] * first_sample[channel];
        }
    }
    state->initialized = 1u;
    return FM_CSSD_LDA_WARMING_UP;
}

fm_cssd_lda_status_t fm_cssd_lda_push_sample(
    fm_cssd_lda_state_t *state,
    const float sample[FM_CSSD_LDA_CHANNELS],
    fm_cssd_lda_output_t *output)
{
    size_t channel;
    const size_t write_index = (state != NULL) ? state->ring_write_index : 0u;

    if (state == NULL || sample == NULL || output == NULL || state->initialized == 0u
        || !fm_is_finite_sample(sample)) {
        return FM_CSSD_LDA_ERROR;
    }

    for (channel = 0u; channel < FM_CSSD_LDA_CHANNELS; ++channel) {
        state->bp_ring[channel][write_index] = fm_filter_one(
            fm_bp_sos,
            FM_CSSD_LDA_BP_SOS_SECTIONS,
            state->bp_filter_state,
            channel,
            sample[channel]);
        state->erd_ring[channel][write_index] = fm_filter_one(
            fm_erd_sos,
            FM_CSSD_LDA_ERD_SOS_SECTIONS,
            state->erd_filter_state,
            channel,
            sample[channel]);
    }

    state->ring_write_index =
        (uint16_t)((write_index + 1u) % FM_CSSD_LDA_HISTORY_SAMPLES);
    if (state->ring_count < FM_CSSD_LDA_HISTORY_SAMPLES) {
        ++state->ring_count;
    }
    if (state->samples_seen < FM_CSSD_LDA_COLD_START_SAMPLES) {
        ++state->samples_seen;
    }
    return fm_predict(state, output);
}

fm_cssd_lda_status_t fm_cssd_lda_push_block(
    fm_cssd_lda_state_t *state,
    const float *samples,
    size_t sample_count,
    fm_cssd_lda_output_t *output)
{
    size_t time;
    fm_cssd_lda_status_t status = FM_CSSD_LDA_WARMING_UP;

    if (state == NULL || samples == NULL || output == NULL || sample_count == 0u) {
        return FM_CSSD_LDA_ERROR;
    }
    for (time = 0u; time < sample_count; ++time) {
        status = fm_cssd_lda_push_sample(
            state, &samples[time * FM_CSSD_LDA_CHANNELS], output);
        if (status == FM_CSSD_LDA_ERROR) {
            return status;
        }
    }
    return status;
}
