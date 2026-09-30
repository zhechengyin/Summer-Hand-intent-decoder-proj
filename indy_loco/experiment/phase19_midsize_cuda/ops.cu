// FP32 only. No fast math, atomics, gate removal or persistent model state.
extern "C" __global__ void residual_forward(
    const float* c, const float* x, float* y, int N, int T, int L) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float p = c[(j / T) * L + j % T] + x[j];
    y[j] = p > 0.0f ? p : 0.0f;
}

extern "C" __global__ void residual_backward(
    const float* y, const float* g, float* dc, float* dx,
    int N, int T, int L) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    int t = j % L, i = (j / L) * T + t;
    float v = t < T && y[i] > 0.0f ? g[i] : 0.0f;
    dc[j] = v;
    if (t < T) dx[i] = v;
}

__device__ float warp_sum(float x) {
    for (int d = 16; d > 0; d >>= 1)
        x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}

// One warp handles one (batch,time) row across 64 channels, in B,C,T layout.
extern "C" __global__ void norm_forward(
    const float* x, const float* weight, const float* bias,
    float* y, float* mean, float* invstd, int B, int T) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int row = thread / 32, lane = thread % 32;
    if (row >= B * T) return;
    int base = (row / T) * 64 * T + row % T;
    int a = base + lane * T, b = a + 32 * T;
    float xa = x[a], xb = x[b];
    float mu = warp_sum(xa + xb) / 64.0f;
    float da = xa - mu, db = xb - mu;
    float variance = warp_sum(da * da + db * db) / 64.0f;
    float inv = 1.0f / sqrtf(variance + 1.0e-5f);
    float ya = (da * inv) * weight[lane] + bias[lane];
    float yb = (db * inv) * weight[lane + 32] + bias[lane + 32];
    y[a] = ya > 0.0f ? ya : 0.0f;
    y[b] = yb > 0.0f ? yb : 0.0f;
    if (lane == 0) { mean[row] = mu; invstd[row] = inv; }
}

extern "C" __global__ void norm_backward(
    const float* x, const float* weight, const float* y,
    const float* mean, const float* invstd, const float* g,
    float* dx, float* dw_partial, float* db_partial, int B, int T) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int row = thread / 32, lane = thread % 32;
    if (row >= B * T) return;
    int base = (row / T) * 64 * T + row % T;
    int a = base + lane * T, b = a + 32 * T;
    float inv = invstd[row];
    float ha = (x[a] - mean[row]) * inv;
    float hb = (x[b] - mean[row]) * inv;
    float ga = y[a] > 0.0f ? g[a] : 0.0f;
    float gb = y[b] > 0.0f ? g[b] : 0.0f;
    float qa = ga * weight[lane], qb = gb * weight[lane + 32];
    float sumq = warp_sum(qa + qb) / 64.0f;
    float sumqh = warp_sum(qa * ha + qb * hb) / 64.0f;
    dx[a] = inv * (qa - sumq - ha * sumqh);
    dx[b] = inv * (qb - sumq - hb * sumqh);
    dw_partial[a] = ga * ha; dw_partial[b] = gb * hb;
    db_partial[a] = ga; db_partial[b] = gb;
}

extern "C" __global__ void gate_forward(
    const float* u, const float* v, const float* previous,
    float* next, float* gates, int B, int H) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= B * H) return;
    int base = (j / H) * 3 * H + j % H;
    float r = 1.0f / (1.0f + expf(-(u[base] + v[base])));
    float z = 1.0f / (1.0f + expf(-(u[base + H] + v[base + H])));
    // Match PyTorch's reset-after-projection convention, including recurrent bias.
    float n = tanhf(u[base + 2 * H] + r * v[base + 2 * H]);
    gates[base] = r; gates[base + H] = z; gates[base + 2 * H] = n;
    next[j] = (1.0f - z) * n + z * previous[j];
}

extern "C" __global__ void gate_backward(
    const float* v, const float* previous, const float* gates, const float* g,
    float* du, float* dv, float* dh, int B, int H) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= B * H) return;
    int base = (j / H) * 3 * H + j % H;
    float r = gates[base], z = gates[base + H], n = gates[base + 2 * H];
    float dn = (g[j] * (1.0f - z)) * (1.0f - n * n);
    float dz = ((g[j] * (previous[j] - n)) * z) * (1.0f - z);
    float dr = ((dn * v[base + 2 * H]) * r) * (1.0f - r);
    du[base] = dv[base] = dr;
    du[base + H] = dv[base + H] = dz;
    du[base + 2 * H] = dn;
    dv[base + 2 * H] = dn * r;
    dh[j] = g[j] * z;
}
