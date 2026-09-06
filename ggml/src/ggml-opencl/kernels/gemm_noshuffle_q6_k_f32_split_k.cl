#ifndef K_SPLITS
#define K_SPLITS 1
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

__attribute__((qcom_reqd_sub_group_size("half")))
kernel void kernel_gemm_noshuffle_q6_k_f32_split_k_4x8(
        global const ushort * src0_ql,
        global const uchar  * src0_qh,
        global const ushort * src0_s,
        global const half   * src0_d,
        read_only image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int m,
        int n,
        int k,
        int n_no_padding,
        ushort mask_f000,
        uchar  mask_c0
) {
    dst = (global float *)((global char *)dst + offsetd);
    const int split = get_global_id(2);
    const int chunk = ((k + K_SPLITS * 16 - 1) / (K_SPLITS * 16)) * 16;
    const int k_start = split * chunk;
    const int k_end = min(k_start + chunk, k);
    dst += split * m * n_no_padding;

    int n_4 = n >> 2;

    int gy = get_global_id(0); // n
    int gx = get_global_id(1); // m
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    global const ushort * ptr_ql = src0_ql + gx_2;
    global const uchar  * ptr_qh = src0_qh + gx_2;
    global const ushort * ptr_s  = src0_s  + gx_2;
    global const half   * ptr_d  = src0_d  + gx_2;

    for (int step = k_start; step < k_end; step += 16) {
        char8 sc8 = as_char8(vload4(0, ptr_s + (step / 32) * m));
        char4 sc = ((step / 16) & 1) == 0 ? sc8.s0246 : sc8.s1357;
        float4 scale = convert_float4(vload4(0, ptr_d + (step / 256) * m)) * convert_float4(sc);
        // Decode each signed scale once for its 16-weight block.
        for (int l = 0; l < 16; l += 4) {
            int i = step + l;
            ushort4 bits4 = vload4(0, ptr_ql + (i/4)*m);
            uchar4 bits2 = vload4(0, ptr_qh + (i/4)*m);
            // j=0
            // load 2x 4 elements of activations on N, corresponding to 8 rows on N
            B.s0123 = read_imageh(src1, gy*2+0 + (i+0)*n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (i+0)*n_4);
            dequantized_weights.s0 = convert_half((convert_float((((bits4.s0 >> 0) & 15) | (((bits2.s0 >> 0) & 3) << 4))) - 32.0f) * scale.s0);
            dequantized_weights.s1 = convert_half((convert_float((((bits4.s1 >> 0) & 15) | (((bits2.s1 >> 0) & 3) << 4))) - 32.0f) * scale.s1);
            dequantized_weights.s2 = convert_half((convert_float((((bits4.s2 >> 0) & 15) | (((bits2.s2 >> 0) & 3) << 4))) - 32.0f) * scale.s2);
            dequantized_weights.s3 = convert_half((convert_float((((bits4.s3 >> 0) & 15) | (((bits2.s3 >> 0) & 3) << 4))) - 32.0f) * scale.s3);
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=1
            B.s0123 = read_imageh(src1, gy*2+0 + (i+1)*n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (i+1)*n_4);
            dequantized_weights.s0 = convert_half((convert_float((((bits4.s0 >> 4) & 15) | (((bits2.s0 >> 2) & 3) << 4))) - 32.0f) * scale.s0);
            dequantized_weights.s1 = convert_half((convert_float((((bits4.s1 >> 4) & 15) | (((bits2.s1 >> 2) & 3) << 4))) - 32.0f) * scale.s1);
            dequantized_weights.s2 = convert_half((convert_float((((bits4.s2 >> 4) & 15) | (((bits2.s2 >> 2) & 3) << 4))) - 32.0f) * scale.s2);
            dequantized_weights.s3 = convert_half((convert_float((((bits4.s3 >> 4) & 15) | (((bits2.s3 >> 2) & 3) << 4))) - 32.0f) * scale.s3);
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=2
            B.s0123 = read_imageh(src1, gy*2+0 + (i+2)*n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (i+2)*n_4);
            dequantized_weights.s0 = convert_half((convert_float((((bits4.s0 >> 8) & 15) | (((bits2.s0 >> 4) & 3) << 4))) - 32.0f) * scale.s0);
            dequantized_weights.s1 = convert_half((convert_float((((bits4.s1 >> 8) & 15) | (((bits2.s1 >> 4) & 3) << 4))) - 32.0f) * scale.s1);
            dequantized_weights.s2 = convert_half((convert_float((((bits4.s2 >> 8) & 15) | (((bits2.s2 >> 4) & 3) << 4))) - 32.0f) * scale.s2);
            dequantized_weights.s3 = convert_half((convert_float((((bits4.s3 >> 8) & 15) | (((bits2.s3 >> 4) & 3) << 4))) - 32.0f) * scale.s3);
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=3
            B.s0123 = read_imageh(src1, gy*2+0 + (i+3)*n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (i+3)*n_4);
            dequantized_weights.s0 = convert_half((convert_float((((bits4.s0 >> 12) & 15) | (((bits2.s0 >> 6) & 3) << 4))) - 32.0f) * scale.s0);
            dequantized_weights.s1 = convert_half((convert_float((((bits4.s1 >> 12) & 15) | (((bits2.s1 >> 6) & 3) << 4))) - 32.0f) * scale.s1);
            dequantized_weights.s2 = convert_half((convert_float((((bits4.s2 >> 12) & 15) | (((bits2.s2 >> 6) & 3) << 4))) - 32.0f) * scale.s2);
            dequantized_weights.s3 = convert_half((convert_float((((bits4.s3 >> 12) & 15) | (((bits2.s3 >> 6) & 3) << 4))) - 32.0f) * scale.s3);
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;
        }

        }
    int idx = gy*8*m + gx_2;
    if (gy*8+0 < n_no_padding) vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst+idx);
    idx += m;
    if (gy*8+1 < n_no_padding) vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst+idx);
    idx += m;
    if (gy*8+2 < n_no_padding) vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst+idx);
    idx += m;
    if (gy*8+3 < n_no_padding) vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst+idx);
    idx += m;
    if (gy*8+4 < n_no_padding) vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst+idx);
    idx += m;
    if (gy*8+5 < n_no_padding) vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst+idx);
    idx += m;
    if (gy*8+6 < n_no_padding) vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst+idx);
    idx += m;
    if (gy*8+7 < n_no_padding) vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst+idx);

}

kernel void kernel_reduce_q6_k_split_k(global const float * partials, global float * dst, ulong offsetd, int count) {
    dst = (global float *)((global char *)dst + offsetd);
    const int i = get_global_id(0) * 4;
    if (i >= count) return;
    float4 sum = 0;
    #pragma unroll
    for (int split = 0; split < K_SPLITS; ++split) {
        sum += vload4(0, partials + split * count + i);
    }
    vstore4(sum, 0, dst + i);
}
