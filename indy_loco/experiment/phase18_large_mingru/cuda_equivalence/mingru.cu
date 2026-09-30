// One lane per sample/feature; coalesced access across features.
// FP32, no fast math or FMA contraction. Retains the positive-candidate
// log-domain prefix formulation, including softplus's default threshold 20.
__device__ float sp(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}
__device__ float dsp(float x) {
    return x > 20.0f ? 1.0f : 1.0f / (1.0f + expf(-x));
}
__device__ float sp_backward(float dy, float x) {
    // Match PyTorch softplus backward's multiply-then-divide evaluation order.
    // dy * sigmoid(x) is mathematically equal, but not FP32 bitwise equal.
    if (x > 20.0f) return dy;
    float z = expf(x);
    return (dy * z) / (z + 1.0f);
}
__device__ float ladd(float a, float b) {
    float m = fmaxf(a, b);
    return m + log1pf(expf(-fabsf(a - b)));
}
// Conservative candidate: fuse elementwise preparation and its backward,
// while retaining PyTorch's exact scan/autograd path for state accumulation.
extern "C" __global__ void mingru_prepare(
    const float* p, float* decay, float* write, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = (i / D) * 2 * D + i % D;
    float a = p[j], b = p[j + D];
    float logg = a >= 0.0f ? logf(a + 0.5f) : -sp(-a);
    decay[i] = -sp(b);
    write[i] = -sp(-b) + logg;
}
extern "C" __global__ void mingru_prepare_backward(
    const float* p, const float* gd, const float* gw, float* gp, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = (i / D) * 2 * D + i % D;
    float a = p[j], b = p[j + D];
    gp[j] = a >= 0.0f ? gw[i] / (a + 0.5f) : sp_backward(gw[i], -a);
    gp[j + D] = sp_backward(-gd[i], b) + sp_backward(gw[i], -b);
}
extern "C" __global__ void mingru_forward(
    const float* projection, float* states, int B, int T, int D) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= B * D) return;
    int n = lane / D, d = lane % D;
    float prefix = 0.0f, total = 0.0f;
    for (int t = 0; t < T; ++t) {
        int p = (n * T + t) * (2 * D) + d;
        float a = projection[p], b = projection[p + D];
        float logg = a >= 0.0f ? logf(a + 0.5f) : -sp(-a);
        prefix = prefix - sp(b);
        float term = -sp(-b) + logg - prefix;
        total = t == 0 ? term : ladd(total, term);
        states[(n * T + t) * D + d] = expf(prefix + total);
    }
}
extern "C" __global__ void mingru_backward(
    const float* projection, const float* states, const float* grad_output,
    float* grad_projection, int B, int T, int D, int last_only) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= B * D) return;
    int n = lane / D, d = lane % D;
    float carry = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        int q = (n * T + t) * D + d;
        int p = (n * T + t) * (2 * D) + d;
        float direct = last_only ? (t == T - 1 ? grad_output[n * D + d] : 0.0f)
                                 : grad_output[q];
        float r = direct + carry;
        float a = projection[p], b = projection[p + D];
        float g = a >= 0.0f ? a + 0.5f : expf(-sp(-a));
        float gp = a >= 0.0f ? 1.0f : g * dsp(-a);
        float decay = expf(-sp(b)), write = expf(-sp(-b));
        float previous = t == 0 ? 0.0f : states[q - D];
        grad_projection[p] = r * write * gp;
        grad_projection[p + D] = r * (write * g * dsp(-b) - decay * previous * dsp(b));
        carry = r * decay;
    }
}
