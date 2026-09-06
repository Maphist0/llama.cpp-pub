#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void flash_attn_sdar_f32_f16_q4(
        __global const uchar * q_data, ulong q_offset,
        __global const uchar * k_data, ulong k_offset,
        __global const uchar * v_data, ulong v_offset,
        __global const uchar * mask_data, ulong mask_offset,
        __global uchar * out_data, ulong out_offset,
        ulong q_nb1, ulong q_nb2, ulong k_nb1, ulong k_nb2,
        ulong v_nb1, ulong v_nb2, ulong mask_nb1,
        int n_kv, int n_head, int n_head_kv, float scale,
        __local float4 * scores, __local float4 * reductions) {
    const int lane = get_local_id(0);
    const int head = get_group_id(1);
    const int token = 4 * get_group_id(2);
    const int kv_head = head / (n_head / n_head_kv);
    __global const float * q0 = (__global const float *)(q_data + q_offset + token * q_nb1 + head * q_nb2);
    __global const float * q1 = (__global const float *)((__global const uchar *)q0 + q_nb1);
    __global const float * q2 = (__global const float *)((__global const uchar *)q1 + q_nb1);
    __global const float * q3 = (__global const float *)((__global const uchar *)q2 + q_nb1);
    float4 maximum = -INFINITY;
    for (int pos = lane; pos < n_kv; pos += 64) {
        float4 bias = 0;
        if (mask_data) {
            bias.s0 = (float)*(__global const half *)(mask_data + mask_offset + (token + 0) * mask_nb1 + pos * 2);
            bias.s1 = (float)*(__global const half *)(mask_data + mask_offset + (token + 1) * mask_nb1 + pos * 2);
            bias.s2 = (float)*(__global const half *)(mask_data + mask_offset + (token + 2) * mask_nb1 + pos * 2);
            bias.s3 = (float)*(__global const half *)(mask_data + mask_offset + (token + 3) * mask_nb1 + pos * 2);
        }
        float4 score = -INFINITY;
        if (any(bias != (float4)(-INFINITY))) {
            __global const half * k = (__global const half *)(k_data + k_offset + pos * k_nb1 + kv_head * k_nb2);
            float4 sum = 0;
            for (int d = 0; d < 128; d += 4) {
                const float4 key = convert_float4(vload4(0, k + d));
                sum += (float4)(dot(vload4(0, q0 + d), key), dot(vload4(0, q1 + d), key),
                                dot(vload4(0, q2 + d), key), dot(vload4(0, q3 + d), key));
            }
            score = sum * scale + bias;
        }
        scores[pos] = score;
        maximum = fmax(maximum, score);
    }
    reductions[lane] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) reductions[lane] = fmax(reductions[lane], reductions[lane + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = reductions[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 denominator = 0;
    for (int pos = lane; pos < n_kv; pos += 64) {
        const float4 w = select((float4)0, exp(scores[pos] - maximum), maximum != (float4)(-INFINITY));
        scores[pos] = w;
        denominator += w;
    }
    reductions[lane] = denominator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) reductions[lane] += reductions[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    denominator = reductions[0];
    float2 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int pos = 0; pos < n_kv; ++pos) {
        const float4 w = scores[pos];
        if (any(w != (float4)0)) {
            __global const half * v = (__global const half *)(v_data + v_offset + pos * v_nb1 + kv_head * v_nb2);
            const float2 value = convert_float2(vload2(0, v + lane * 2));
            a0 += w.s0 * value;
            a1 += w.s1 * value;
            a2 += w.s2 * value;
            a3 += w.s3 * value;
        }
    }
    __global float * out = (__global float *)(out_data + out_offset) + (token * n_head + head) * 128 + lane * 2;
    vstore2(denominator.s0 > 0 ? a0 / denominator.s0 : (float2)0, 0, out);
    vstore2(denominator.s1 > 0 ? a1 / denominator.s1 : (float2)0, 0, out + n_head * 128);
    vstore2(denominator.s2 > 0 ? a2 / denominator.s2 : (float2)0, 0, out + n_head * 256);
    vstore2(denominator.s3 > 0 ? a3 / denominator.s3 : (float2)0, 0, out + n_head * 384);
}

__kernel void flash_attn_sdar_f32_f16(
        __global const uchar * q_data, ulong q_offset,
        __global const uchar * k_data, ulong k_offset,
        __global const uchar * v_data, ulong v_offset,
        __global const uchar * mask_data, ulong mask_offset,
        __global uchar * out_data, ulong out_offset,
        ulong q_nb1, ulong q_nb2, ulong k_nb1, ulong k_nb2,
        ulong v_nb1, ulong v_nb2, ulong mask_nb1,
        int n_kv, int n_head, int n_head_kv, float scale,
        __local float * scores, __local float * reductions) {
    const int lane = get_local_id(0);
    const int head = get_group_id(1);
    const int token = get_group_id(2);
    const int kv_head = head / (n_head / n_head_kv);
    __global const float * q = (__global const float *)(q_data + q_offset + token * q_nb1 + head * q_nb2);
    __global const half * mask = mask_data ? (__global const half *)(mask_data + mask_offset + token * mask_nb1) : 0;
    float max_score = -INFINITY;
    for (int pos = lane; pos < n_kv; pos += 64) {
        const float bias = mask ? (float)mask[pos] : 0.0f;
        float score = -INFINITY;
        if (bias != -INFINITY) {
            __global const half * k = (__global const half *)(k_data + k_offset + pos * k_nb1 + kv_head * k_nb2);
            float sum = 0;
            for (int d = 0; d < 128; d += 4) {
                sum += dot(vload4(0, q + d), convert_float4(vload4(0, k + d)));
            }
            score = sum * scale + bias;
        }
        scores[pos] = score;
        max_score = fmax(max_score, score);
    }
    reductions[lane] = max_score;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) reductions[lane] = fmax(reductions[lane], reductions[lane + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    max_score = reductions[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float denominator = 0;
    for (int pos = lane; pos < n_kv; pos += 64) {
        const float weight = max_score == -INFINITY ? 0.0f : exp(scores[pos] - max_score);
        scores[pos] = weight;
        denominator += weight;
    }
    reductions[lane] = denominator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) reductions[lane] += reductions[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    denominator = reductions[0];
    float2 sum = 0;
    for (int pos = 0; pos < n_kv; ++pos) {
        if (scores[pos] != 0.0f) {
            __global const half * v = (__global const half *)(v_data + v_offset + pos * v_nb1 + kv_head * v_nb2);
            sum += scores[pos] * convert_float2(vload2(0, v + lane * 2));
        }
    }
    __global float * out = (__global float *)(out_data + out_offset);
    vstore2(denominator > 0.0f ? sum / denominator : (float2)0, 0, out + (token * n_head + head) * 128 + lane * 2);
}
