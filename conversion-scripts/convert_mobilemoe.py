#!/usr/bin/env python3
"""Convert MobileMoE S/M/L QAT directly to GGUF; exact INT4 is the default.

Example:
  python convert_mobilemoe.py --source /path/to/MobileMoE-S-QAT \
    --llama /path/to/extended/llama.cpp --output model.gguf

--outtype f32 exports the decoded baseline. --outtype requantized-int4
refits only the transposed experts for the comparison control. Exact INT4
preserves every original code/scale pair and verifies all checkpoint payloads
by reading the completed GGUF back. No intermediate GGUF or helper scripts.
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def unpack(p):
    q = np.stack((p & 15, p >> 4), axis=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)
    return ((q.astype(np.int16) + 8) % 16 - 8).astype(np.int8)


def pack_q40(q, scales):
    codes = (q.reshape(-1, 32).astype(np.int16) + 8).astype(np.uint8)
    d = np.ascontiguousarray(scales, dtype='<f2').reshape(-1, 1).view(np.uint8)
    b = codes[:, :16] | (codes[:, 16:] << 4)
    return np.concatenate((d, b), axis=1).reshape(*q.shape[:-1], q.shape[-1] // 32 * 18)


def pack_q41(d, m, q):
    d = np.ascontiguousarray(d, dtype='<f2').reshape(-1, 1).view(np.uint8)
    m = np.ascontiguousarray(m, dtype='<f2').reshape(-1, 1).view(np.uint8)
    return np.concatenate((d, m, q[:, :16] | (q[:, 16:] << 4)), axis=1)


def fit_affine(x, gguf):
    """Retain the lowest actual F16-scale reconstruction error per block."""
    shape = x.shape
    blocks = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    output = np.empty((len(blocks), 20), dtype=np.uint8)
    for begin in range(0, len(blocks), 16384):
        z = blocks[begin:begin + 16384]
        standard = gguf.quants.quantize(z, gguf.GGMLQuantizationType.Q4_1).reshape(-1, 20)
        decoded = gguf.quants.dequantize(standard, gguf.GGMLQuantizationType.Q4_1).reshape(-1, 32)
        best_err = np.sum((decoded - z) ** 2, axis=1)
        best = standard.copy()
        lo = z.min(axis=1)
        hi = z.max(axis=1)
        center = (hi + lo) * 0.5
        nominal = (hi - lo) / 15
        for fraction in (0.85, 0.925, 1.0, 1.075):
            d = nominal * fraction
            m = center - 7.5 * d
            for _ in range(5):
                # Include stored F16 rounding when selecting codes and candidates.
                d = d.astype(np.float16).astype(np.float32)
                m = m.astype(np.float16).astype(np.float32)
                inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
                q = np.clip(np.rint((z - m[:, None]) * inv[:, None]), 0, 15).astype(np.uint8)
                reconstructed = d[:, None] * q + m[:, None]
                err = np.sum((reconstructed - z) ** 2, axis=1)
                take = err < best_err
                if take.any():
                    best[take] = pack_q41(d[take], m[take], q[take])
                    best_err[take] = err[take]
                qf = q.astype(np.float32)
                mean_q = qf.mean(axis=1)
                mean_z = z.mean(axis=1)
                centered_q = qf - mean_q[:, None]
                denominator = np.sum(centered_q ** 2, axis=1)
                numerator = np.sum(centered_q * (z - mean_z[:, None]), axis=1)
                d = np.divide(numerator, denominator, out=d.copy(), where=denominator > 0)
                m = mean_z - d * mean_q
        output[begin:begin + len(z)] = best
    return output.reshape(*shape[:-1], shape[-1] // 32 * 20)


def verify_exact(path, source_dir, config, gguf):
    reader = gguf.GGUFReader(path)
    assert reader.fields['general.architecture'].contents() == 'mobilemoe'
    assert reader.fields['mobilemoe.expert_layout'].contents() == 'qat_output_grouped_g32'
    tensors = {t.name: t for t in reader.tensors}
    results = []

    def decode(name):
        t = tensors[name]
        assert t.tensor_type == gguf.GGMLQuantizationType.Q4_0
        shape = tuple(reversed(t.shape.tolist()))
        blocks = t.data.reshape(-1, 18)
        scale = blocks[:, :2].copy().view('<f2').reshape(*shape[:-1], shape[-1] // 32)
        unsigned = np.concatenate((blocks[:, 2:] & 15, blocks[:, 2:] >> 4), axis=1)
        return (unsigned.astype(np.int16) - 8).reshape(shape), scale

    with safe_open(source_dir / 'model.safetensors', framework='pt', device='cpu') as source:
        def verify(key, data):
            original = source.get_tensor(key)
            raw = original.contiguous().view(torch.uint8).numpy().tobytes()
            candidate = np.ascontiguousarray(data).tobytes()
            assert candidate == raw, key
            results.append({'name': key, 'bytes': len(raw), 'sha256': hashlib.sha256(candidate).hexdigest(), 'byte_exact': True})

        def pair(key, q, scale):
            # Reconstruct the checkpoint's adjacent signed two's-complement nibbles.
            nibble = (q & 15).astype(np.uint8)
            packed = nibble[..., ::2] | (nibble[..., 1::2] << 4)
            verify(key + '.qweight', packed)
            verify(key + '.weight_scale', scale)

        def dense(key, name, heads=None):
            q, scale = decode(name)
            if heads:
                q = q.reshape(heads, 2, config['head_dim'] // 2, -1).transpose(0, 2, 1, 3).reshape(q.shape)
                scale = scale.reshape(heads, 2, config['head_dim'] // 2, -1).transpose(0, 2, 1, 3).reshape(scale.shape)
            pair(key, q, scale)

        def scalar(key, name):
            original = source.get_tensor(key)
            data = torch.from_numpy(tensors[name].data.copy()).to(original.dtype)
            verify(key, data.contiguous().view(torch.uint8).numpy())

        dense('model.embed_tokens', 'token_embd.weight')
        eq, es = decode('token_embd.weight')
        oq, os = decode('output.weight')
        assert np.array_equal(eq, oq) and es.tobytes() == os.tobytes()
        del eq, es, oq, os
        scalar('model.norm.weight', 'output_norm.weight')
        for i in range(config['num_hidden_layers']):
            p = f'model.layers.{i}.'
            b = f'blk.{i}.'
            scalar(p + 'input_layernorm.weight', b + 'attn_norm.weight')
            scalar(p + 'post_attention_layernorm.weight', b + 'ffn_norm.weight')
            for part, name, heads in [('q', 'attn_q', config['num_attention_heads']), ('k', 'attn_k', config['num_key_value_heads']), ('v', 'attn_v', None), ('o', 'attn_output', None)]:
                dense(p + f'self_attn.{part}_proj', b + name + '.weight', heads)
            p += 'feed_forward.'
            scalar(p + 'router.weight', b + 'ffn_gate_inp.weight')
            scalar(p + 'expert_bias', b + 'exp_probs_b.bias')
            for part in ('gate', 'up', 'down'):
                dense(p + f'shared_expert.{part}_proj', b + f'ffn_{part}_shexp.weight')
            gq, gs = decode(b + 'ffn_gate_exps.weight')
            uq, us = decode(b + 'ffn_up_exps.weight')
            q = np.concatenate((gq, uq), axis=-1)
            s = np.concatenate((gs, us), axis=-1)
            nibble = (q & 15).astype(np.uint8)
            verify(p + 'experts.gate_up_qweight', nibble[..., ::2] | (nibble[..., 1::2] << 4))
            verify(p + 'experts.gate_up_scale', s)
            dq, ds = decode(b + 'ffn_down_exps.weight')
            nibble = (dq & 15).astype(np.uint8)
            verify(p + 'experts.down_qweight', nibble[..., ::2] | (nibble[..., 1::2] << 4))
            verify(p + 'experts.down_scale', ds)
        assert {r['name'] for r in results} == set(source.keys())
        assert len(results) == len(source.keys())
    report = {'passed': True, 'source_tensors_checked': len(results), 'all_original_tensor_payloads_byte_exact_after_inverse_packing': True,
              'code_and_scale_tensors_checked': sum('qweight' in r['name'] or 'scale' in r['name'] for r in results),
              'gguf_bytes': path.stat().st_size, 'tensors': results}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--llama', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outtype', choices=('exact-int4', 'f32', 'requantized-int4'), default='exact-int4')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.llama), str(args.llama / 'gguf-py')]
    import gguf
    import gguf.quants
    from conversion.base import TextModel

    class VocabModel(TextModel):
        model_arch = gguf.MODEL_ARCH.MOBILEMOE

    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO)
    c = json.loads((args.source / 'config.json').read_text())
    assert c['model_type'] == 'mobilemoe'
    assert all(c['no_rope_layers']) and not c['attn_temperature_tuning']
    assert c['interleave_moe_layer_step'] == 1 and c['use_qk_norm']
    assert c['tie_word_embeddings'] and not c['attention_bias']
    assert c['quantization']['group_size'] == 32
    exact = args.outtype == 'exact-int4'
    file_type = {'exact-int4': gguf.LlamaFileType.MOSTLY_Q4_0, 'f32': gguf.LlamaFileType.ALL_F32,
                 'requantized-int4': gguf.LlamaFileType.MOSTLY_Q4_1}[args.outtype]
    vocab = VocabModel(args.source, file_type, args.output)
    w = gguf.GGUFWriter(args.output, 'mobilemoe', use_temp_file=True)
    vocab.gguf_writer = w
    vocab.set_vocab()
    w.add_name(args.source.name + ' ' + args.outtype)
    w.add_string('mobilemoe.expert_layout', 'qat_output_grouped_g32' if exact else 'input_grouped')
    w.add_string('mobilemoe.source_architecture', 'MobileMoEForCausalLM')
    w.add_string('mobilemoe.weight_materialization', {
        'exact-int4': 'Original INT4 codes and FP16 scales; BF16 expert materialization; no requantization',
        'f32': 'INT4/g32 -> BF16 -> F32; routers remain F32',
        'requantized-int4': 'Original aligned Q4_0 groups; transposed experts fitted as Q4_1',
    }[args.outtype])
    if args.outtype != 'f32':
        w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_string('mobilemoe.compatibility', 'All layers use RoPE; window covers the entire 8192-token context. QK RMS normalization commutes with rotation in real arithmetic.')
    w.add_file_type(file_type)
    for fn, value in (
        ('vocab_size', c['vocab_size']), ('context_length', c['max_position_embeddings']),
        ('embedding_length', c['hidden_size']), ('block_count', c['num_hidden_layers']),
        ('feed_forward_length', c['intermediate_size_mlp']),
        ('expert_feed_forward_length', c['intermediate_size']),
        ('expert_shared_feed_forward_length', c['intermediate_size_mlp']),
        ('head_count', c['num_attention_heads']), ('head_count_kv', c['num_key_value_heads']),
        ('key_length', c['head_dim']), ('value_length', c['head_dim']),
        ('rope_dimension_count', c['head_dim']), ('rope_freq_base', c['rope_theta']),
        ('layer_norm_rms_eps', c['rms_norm_eps']), ('expert_count', c['num_local_experts']),
        ('expert_used_count', c['num_experts_per_tok']), ('expert_shared_count', 1),
        ('expert_weights_scale', c['routed_scaling_factor']),
        ('expert_weights_norm', c['norm_topk_prob']),
        ('expert_gating_func', gguf.ExpertGatingFuncType.SIGMOID),
        ('sliding_window', c['max_position_embeddings']), ('sliding_window_pattern', 0),
        ('rope_scaling_type', gguf.RopeScalingType.NONE),
        ('bos_token_id', c['bos_token_id']), ('eos_token_id', 128001),
        ('eot_token_id', 128008), ('eom_token_id', 128007),
    ):
        getattr(w, 'add_' + fn)(value)

    rs = c['rope_scaling']
    inv = 1 / c['rope_theta'] ** (torch.arange(0, c['head_dim'], 2).float() / c['head_dim'])
    factors = []
    for freq in inv:
        wavelength = 2 * torch.pi / freq
        if wavelength < rs['original_max_position_embeddings'] / rs['high_freq_factor']:
            factor = 1.0
        elif wavelength > rs['original_max_position_embeddings'] / rs['low_freq_factor']:
            factor = rs['factor']
        elif rs['high_freq_factor'] == rs['low_freq_factor']:
            factor = 1.0
        else:
            smooth = (rs['original_max_position_embeddings'] / wavelength - rs['low_freq_factor']) / (rs['high_freq_factor'] - rs['low_freq_factor'])
            factor = 1 / ((1 - smooth) / rs['factor'] + smooth)
        factors.append(float(factor))

    manifest = []
    writing = False
    def add(name, data, typ=None):
        if isinstance(data, torch.Tensor):
            data = data.float().numpy()
        data = np.ascontiguousarray(data)
        assert np.isfinite(data).all(), name
        if args.outtype != 'f32':
            w.add_tensor(name, data, raw_dtype=typ)
        elif writing:
            w.write_tensor_data(data)
        else:
            w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=typ)
        if not writing:
            manifest.append({'name': name, 'type': typ.name if typ else 'F32'})

    def emit(name, q, scale, expert=False):
        if args.outtype == 'f32' or (expert and not exact):
            shape = q.shape
            x = torch.from_numpy(np.ascontiguousarray(q)).reshape(*shape[:-1], shape[-1] // 32, 32).float()
            s = torch.from_numpy(np.ascontiguousarray(scale)).float().unsqueeze(-1)
            data = (x * s).reshape(shape).to(torch.bfloat16).float().numpy()
            if expert:
                data = np.ascontiguousarray(data.swapaxes(-1, -2))
            if args.outtype == 'f32':
                add(name, data)
            else:
                add(name, fit_affine(data, gguf), gguf.GGMLQuantizationType.Q4_1)
        else:
            add(name, pack_q40(q, scale), gguf.GGMLQuantizationType.Q4_0)

    def permute_rows(data, heads):
        return data.reshape(heads, c['head_dim'] // 2, 2, -1).transpose(0, 2, 1, 3).reshape(data.shape)

    def convert_weights():
        add('rope_freqs.weight', torch.tensor(factors))
        with safe_open(args.source / 'model.safetensors', framework='pt', device='cpu') as f:
            consumed = set()
            def get(name):
                consumed.add(name)
                return f.get_tensor(name)
            def pair(base, q_suffix='.qweight', s_suffix='.weight_scale'):
                q = unpack(get(base + q_suffix).numpy())
                scale = get(base + s_suffix)
                assert scale.dtype == torch.float16, base
                return q, scale.numpy()

            q, scale = pair('model.embed_tokens')
            emit('token_embd.weight', q, scale)
            emit('output.weight', q, scale)
            del q, scale
            add('output_norm.weight', get('model.norm.weight'))
            for i in range(c['num_hidden_layers']):
                p = f'model.layers.{i}.'
                b = f'blk.{i}.'
                add(b + 'attn_norm.weight', get(p + 'input_layernorm.weight'))
                add(b + 'ffn_norm.weight', get(p + 'post_attention_layernorm.weight'))
                for part in ('q', 'k', 'v', 'o'):
                    q, scale = pair(p + 'self_attn.' + part + '_proj')
                    if part in ('q', 'k'):
                        heads = c['num_attention_heads'] if part == 'q' else c['num_key_value_heads']
                        q, scale = permute_rows(q, heads), permute_rows(scale, heads)
                    emit(b + ('attn_output' if part == 'o' else 'attn_' + part) + '.weight', q, scale)
                for part in ('q', 'k'):
                    add(b + 'attn_' + part + '_norm.weight', torch.ones(c['head_dim']))
                p += 'feed_forward.'
                add(b + 'ffn_gate_inp.weight', get(p + 'router.weight'))
                add(b + 'exp_probs_b.bias', get(p + 'expert_bias'))
                q, scale = pair(p + 'experts.gate_up', '_qweight', '_scale')
                gq, uq = np.split(q, 2, axis=-1)
                gs, us = np.split(scale, 2, axis=-1)
                emit(b + 'ffn_gate_exps.weight', gq, gs, expert=True)
                emit(b + 'ffn_up_exps.weight', uq, us, expert=True)
                q, scale = pair(p + 'experts.down', '_qweight', '_scale')
                emit(b + 'ffn_down_exps.weight', q, scale, expert=True)
                for part in ('gate', 'up', 'down'):
                    q, scale = pair(p + 'shared_expert.' + part + '_proj')
                    emit(b + 'ffn_' + part + '_shexp.weight', q, scale)
                print(f'converted layer {i}', flush=True)
            assert consumed == set(f.keys()), sorted(set(f.keys()) - consumed)
        return consumed

    consumed = convert_weights()
    w.write_header_to_file()
    w.write_kv_data_to_file()
    if args.outtype == 'f32':
        # Stream a second pass to avoid storing two full F32 copies on disk.
        w.write_ti_data_to_file()
        writing = True
        assert convert_weights() == consumed
        assert not any(w.tensors)
    else:
        w.write_tensors_to_file(progress=True)
    w.close()
    report = {'source': str(args.source), 'source_tensors': len(consumed), 'architecture': 'mobilemoe',
              'outtype': args.outtype, 'requantizes_experts': args.outtype == 'requantized-int4', 'tensors': manifest}
    if exact:
        report['verification'] = verify_exact(args.output, args.source, c, gguf)
        print(f"verified {report['verification']['source_tensors_checked']} original tensor payloads", flush=True)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(args.output, args.output.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
