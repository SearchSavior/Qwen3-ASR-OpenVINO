#!/usr/bin/env python3
"""
OpenVINO conversion script for Qwen3-ASR (0.6B / 1.7B).

Converts the PyTorch model into two OpenVINO IR sub-models:
  1. audio_encoder  – mel spectrogram [1, 128, T] -> audio embeddings [1, N, output_dim]
  2. decoder        – unified prefill / decode with KV-cache

Usage:
    pip install torch safetensors openvino numpy nncf
    python qwen3_asr_ov_convert.py <model_dir> [--ov-dir ov_model]
"""

import os
import json
import math
import argparse
import shutil
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors import safe_open


NUM_MEL_BINS = 128
EXTRA_MODEL_FILES = [
    "chat_template.json",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "vocab.json",
    "tokenizer_config.json",
]


def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg["thinker_config"]
    ac = tc["audio_config"]
    txc = tc["text_config"]
    return {
        "enc_d_model": ac["d_model"],
        "enc_layers": ac["encoder_layers"],
        "enc_heads": ac["encoder_attention_heads"],
        "enc_ffn_dim": ac["encoder_ffn_dim"],
        "enc_output_dim": ac["output_dim"],
        "enc_downsample_hidden": ac["downsample_hidden_size"],
        "enc_num_mel_bins": ac["num_mel_bins"],
        "enc_max_source_pos": ac["max_source_positions"],
        "enc_n_window": ac["n_window"],
        "enc_n_window_infer": ac["n_window_infer"],
        "enc_conv_chunksize": ac.get("conv_chunksize", 500),
        "dec_hidden_size": txc["hidden_size"],
        "dec_layers": txc["num_hidden_layers"],
        "dec_heads": txc["num_attention_heads"],
        "dec_kv_heads": txc["num_key_value_heads"],
        "dec_head_dim": txc["head_dim"],
        "dec_intermediate": txc["intermediate_size"],
        "dec_rms_norm_eps": txc["rms_norm_eps"],
        "dec_rope_theta": txc["rope_theta"],
        "dec_mrope_section": txc["rope_scaling"]["mrope_section"],
        "dec_vocab_size": txc["vocab_size"],
        "audio_start_token_id": tc["audio_start_token_id"],
        "audio_end_token_id": tc["audio_end_token_id"],
        "audio_token_id": tc["audio_token_id"],
    }


class MultiSafetensors:
    def __init__(self, model_dir):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            self.files = {}
            for shard in shard_files:
                self.files[shard] = safe_open(
                    os.path.join(model_dir, shard), framework="pt"
                )
            self.weight_map = index["weight_map"]
        else:
            self.files = {
                "model.safetensors": safe_open(single_path, framework="pt")
            }
            self.weight_map = None

    def get_tensor(self, name):
        if self.weight_map:
            shard = self.weight_map[name]
            return self.files[shard].get_tensor(name)
        for sf_handle in self.files.values():
            try:
                return sf_handle.get_tensor(name)
            except Exception:
                continue
        raise KeyError(f"Weight not found: {name}")


def _w(sf, name):
    t = sf.get_tensor(name)
    return t.float() if t.dtype == torch.bfloat16 else t


def sinusoidal_position_embedding(length, channels, max_timescale=10000):
    log_inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-log_inc * torch.arange(channels // 2).float())
    scaled = torch.arange(length).float().unsqueeze(1) * inv.unsqueeze(0)
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


def compute_rope_freqs(positions, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos(), emb.sin()


class AudioEncoderModule(nn.Module):
    """
    Input:  mel [1, 128, T]  – T MUST be pre-padded to a multiple of chunk_size
    Output: audio_embeds [1, total_tokens, output_dim]
    """

    def __init__(self, sf, cfg):
        super().__init__()
        prefix = "thinker.audio_tower"
        d = cfg["enc_d_model"]
        self.d_model = d
        self.n_layers = cfg["enc_layers"]
        self.n_heads = cfg["enc_heads"]
        self.head_dim = d // self.n_heads
        self.ffn_dim = cfg["enc_ffn_dim"]
        self.n_window = cfg["enc_n_window"]
        self.n_window_infer = cfg["enc_n_window_infer"]
        self.chunk_size = self.n_window * 2

        dh = cfg["enc_downsample_hidden"]
        self.conv1 = nn.Conv2d(1, dh, 3, stride=2, padding=1)
        self.conv1.weight.data.copy_(_w(sf, f"{prefix}.conv2d1.weight"))
        self.conv1.bias.data.copy_(_w(sf, f"{prefix}.conv2d1.bias"))
        self.conv2 = nn.Conv2d(dh, dh, 3, stride=2, padding=1)
        self.conv2.weight.data.copy_(_w(sf, f"{prefix}.conv2d2.weight"))
        self.conv2.bias.data.copy_(_w(sf, f"{prefix}.conv2d2.bias"))
        self.conv3 = nn.Conv2d(dh, dh, 3, stride=2, padding=1)
        self.conv3.weight.data.copy_(_w(sf, f"{prefix}.conv2d3.weight"))
        self.conv3.bias.data.copy_(_w(sf, f"{prefix}.conv2d3.bias"))

        self.conv_out = nn.Linear(dh * (NUM_MEL_BINS // 8), d, bias=False)
        self.conv_out.weight.data.copy_(_w(sf, f"{prefix}.conv_out.weight"))

        tpc = self.chunk_size
        for _ in range(3):
            tpc = (tpc + 1) // 2
        self.tokens_per_chunk = tpc
        self.register_buffer(
            "pos_emb",
            sinusoidal_position_embedding(self.tokens_per_chunk, d),
        )

        self.attn_ln_w = nn.ParameterList()
        self.attn_ln_b = nn.ParameterList()
        self.q_proj_w = nn.ParameterList()
        self.q_proj_b = nn.ParameterList()
        self.k_proj_w = nn.ParameterList()
        self.k_proj_b = nn.ParameterList()
        self.v_proj_w = nn.ParameterList()
        self.v_proj_b = nn.ParameterList()
        self.out_proj_w = nn.ParameterList()
        self.out_proj_b = nn.ParameterList()
        self.ffn_ln_w = nn.ParameterList()
        self.ffn_ln_b = nn.ParameterList()
        self.fc1_w = nn.ParameterList()
        self.fc1_b = nn.ParameterList()
        self.fc2_w = nn.ParameterList()
        self.fc2_b = nn.ParameterList()

        for i in range(self.n_layers):
            lp = f"{prefix}.layers.{i}"
            self.attn_ln_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn_layer_norm.weight")))
            self.attn_ln_b.append(nn.Parameter(_w(sf, f"{lp}.self_attn_layer_norm.bias")))
            self.q_proj_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn.q_proj.weight")))
            self.q_proj_b.append(nn.Parameter(_w(sf, f"{lp}.self_attn.q_proj.bias")))
            self.k_proj_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn.k_proj.weight")))
            self.k_proj_b.append(nn.Parameter(_w(sf, f"{lp}.self_attn.k_proj.bias")))
            self.v_proj_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn.v_proj.weight")))
            self.v_proj_b.append(nn.Parameter(_w(sf, f"{lp}.self_attn.v_proj.bias")))
            self.out_proj_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn.out_proj.weight")))
            self.out_proj_b.append(nn.Parameter(_w(sf, f"{lp}.self_attn.out_proj.bias")))
            self.ffn_ln_w.append(nn.Parameter(_w(sf, f"{lp}.final_layer_norm.weight")))
            self.ffn_ln_b.append(nn.Parameter(_w(sf, f"{lp}.final_layer_norm.bias")))
            self.fc1_w.append(nn.Parameter(_w(sf, f"{lp}.fc1.weight")))
            self.fc1_b.append(nn.Parameter(_w(sf, f"{lp}.fc1.bias")))
            self.fc2_w.append(nn.Parameter(_w(sf, f"{lp}.fc2.weight")))
            self.fc2_b.append(nn.Parameter(_w(sf, f"{lp}.fc2.bias")))

        self.ln_post_w = nn.Parameter(_w(sf, f"{prefix}.ln_post.weight"))
        self.ln_post_b = nn.Parameter(_w(sf, f"{prefix}.ln_post.bias"))

        p1w = _w(sf, f"{prefix}.proj1.weight")
        p1b = _w(sf, f"{prefix}.proj1.bias")
        p2w = _w(sf, f"{prefix}.proj2.weight")
        p2b = _w(sf, f"{prefix}.proj2.bias")
        self.proj1 = nn.Linear(p1w.shape[1], p1w.shape[0])
        self.proj1.weight.data.copy_(p1w)
        self.proj1.bias.data.copy_(p1b)
        self.proj2 = nn.Linear(p2w.shape[1], p2w.shape[0])
        self.proj2.weight.data.copy_(p2w)
        self.proj2.bias.data.copy_(p2b)

    def forward(self, mel):
        cs = self.chunk_size
        mel_4d = mel.reshape(1, NUM_MEL_BINS, -1, cs)
        mel_4d = mel_4d.permute(2, 0, 1, 3)
        mel_4d = mel_4d.reshape(-1, 1, NUM_MEL_BINS, cs)

        x = F.gelu(self.conv1(mel_4d))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        nc, c, f, tpc = x.shape
        x = x.permute(0, 3, 1, 2).reshape(nc, tpc, c * f)

        x = self.conv_out(x)
        x = x + self.pos_emb.unsqueeze(0)
        x = x.reshape(1, nc * tpc, self.d_model)
        n_tokens = x.shape[1]

        tpw = tpc * (self.n_window_infer // self.chunk_size)
        idx = torch.arange(n_tokens, device=x.device)
        win_id = idx // tpw
        mask_bool = win_id.unsqueeze(0) != win_id.unsqueeze(1)
        attn_mask = torch.where(
            mask_bool,
            torch.tensor(-1e9, device=x.device),
            torch.tensor(0.0, device=x.device),
        ).unsqueeze(0).unsqueeze(0)

        nh, hd = self.n_heads, self.head_dim
        for i in range(self.n_layers):
            xn = F.layer_norm(x, (self.d_model,), self.attn_ln_w[i], self.attn_ln_b[i])
            q = F.linear(xn, self.q_proj_w[i], self.q_proj_b[i])
            k = F.linear(xn, self.k_proj_w[i], self.k_proj_b[i])
            v = F.linear(xn, self.v_proj_w[i], self.v_proj_b[i])

            q = q.view(1, n_tokens, nh, hd).transpose(1, 2)
            k = k.view(1, n_tokens, nh, hd).transpose(1, 2)
            v = v.view(1, n_tokens, nh, hd).transpose(1, 2)

            attn = F.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=attn_mask,
                scale=1.0 / math.sqrt(hd),
            )
            attn = attn.transpose(1, 2).contiguous().view(1, n_tokens, nh * hd)
            x = x + F.linear(attn, self.out_proj_w[i], self.out_proj_b[i])

            xn = F.layer_norm(x, (self.d_model,), self.ffn_ln_w[i], self.ffn_ln_b[i])
            x = x + F.linear(
                F.gelu(F.linear(xn, self.fc1_w[i], self.fc1_b[i])),
                self.fc2_w[i],
                self.fc2_b[i],
            )

        x = F.layer_norm(x, (self.d_model,), self.ln_post_w, self.ln_post_b)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x


class ThinkerEmbeddingModule(nn.Module):
    def __init__(self, sf):
        super().__init__()
        embed_w = _w(sf, "thinker.model.embed_tokens.weight")
        vocab, hidden = embed_w.shape
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.embed_tokens.weight.data.copy_(embed_w)

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class DecoderModule(nn.Module):
    def __init__(self, sf, cfg):
        super().__init__()
        self.h = cfg["dec_hidden_size"]
        self.layers = cfg["dec_layers"]
        self.nh = cfg["dec_heads"]
        self.nkv = cfg["dec_kv_heads"]
        self.hd = cfg["dec_head_dim"]
        self.eps = cfg["dec_rms_norm_eps"]
        self.theta = cfg["dec_rope_theta"]
        self.vocab = cfg["dec_vocab_size"]
        self.gqa = self.nh // self.nkv

        self.lm_head = nn.Linear(self.h, self.vocab, bias=False)
        self.lm_head.weight.data.copy_(_w(sf, "thinker.lm_head.weight"))
        self.final_norm = nn.Parameter(_w(sf, "thinker.model.norm.weight"))

        self.in_ln = nn.ParameterList()
        self.post_ln = nn.ParameterList()
        self.q_w = nn.ParameterList()
        self.k_w = nn.ParameterList()
        self.v_w = nn.ParameterList()
        self.o_w = nn.ParameterList()
        self.q_norm = nn.ParameterList()
        self.k_norm = nn.ParameterList()
        self.gate_w = nn.ParameterList()
        self.up_w = nn.ParameterList()
        self.down_w = nn.ParameterList()

        for i in range(self.layers):
            lp = f"thinker.model.layers.{i}"
            self.in_ln.append(nn.Parameter(_w(sf, f"{lp}.input_layernorm.weight")))
            self.post_ln.append(nn.Parameter(_w(sf, f"{lp}.post_attention_layernorm.weight")))
            self.q_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn.q_proj.weight")))
            self.k_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn.k_proj.weight")))
            self.v_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn.v_proj.weight")))
            self.o_w.append(nn.Parameter(_w(sf, f"{lp}.self_attn.o_proj.weight")))
            self.q_norm.append(nn.Parameter(_w(sf, f"{lp}.self_attn.q_norm.weight")))
            self.k_norm.append(nn.Parameter(_w(sf, f"{lp}.self_attn.k_norm.weight")))
            self.gate_w.append(nn.Parameter(_w(sf, f"{lp}.mlp.gate_proj.weight")))
            self.up_w.append(nn.Parameter(_w(sf, f"{lp}.mlp.up_proj.weight")))
            self.down_w.append(nn.Parameter(_w(sf, f"{lp}.mlp.down_proj.weight")))

    @staticmethod
    def _rms(x, w, eps):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (w * (x.float() * torch.rsqrt(var + eps))).to(x.dtype)

    def forward(self, input_embeds, position_ids, *past_kv):
        batch_size, seq_len, _ = input_embeds.shape
        layer_count = self.layers
        past_keys = list(past_kv[:layer_count])
        past_vals = list(past_kv[layer_count:])
        past_len = past_keys[0].shape[2]
        total_kv = past_len + seq_len

        positions = position_ids.squeeze(0)
        rope_cos, rope_sin = compute_rope_freqs(positions, self.hd, self.theta)
        rc = rope_cos.unsqueeze(0).unsqueeze(2)
        rs = rope_sin.unsqueeze(0).unsqueeze(2)
        half = self.hd // 2

        q_pos = position_ids.float()
        kv_pos = torch.arange(total_kv, device=input_embeds.device).float()
        mask_bool = kv_pos.unsqueeze(0) > q_pos.unsqueeze(-1)
        attn_mask = torch.where(
            mask_bool,
            torch.tensor(-1e9, device=input_embeds.device),
            torch.tensor(0.0, device=input_embeds.device),
        ).unsqueeze(1)

        h = input_embeds
        new_keys = []
        new_vals = []

        for i in range(layer_count):
            xn = self._rms(h, self.in_ln[i], self.eps)

            q = F.linear(xn, self.q_w[i]).view(batch_size, seq_len, self.nh, self.hd)
            k = F.linear(xn, self.k_w[i]).view(batch_size, seq_len, self.nkv, self.hd)
            v = F.linear(xn, self.v_w[i]).view(batch_size, seq_len, self.nkv, self.hd)

            q = self._rms(q, self.q_norm[i], self.eps)
            k = self._rms(k, self.k_norm[i], self.eps)

            q = q * rc + torch.cat([-q[..., half:], q[..., :half]], -1) * rs
            k = k * rc + torch.cat([-k[..., half:], k[..., :half]], -1) * rs

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            k_full = torch.cat([past_keys[i], k], dim=2)
            v_full = torch.cat([past_vals[i], v], dim=2)
            new_keys.append(k_full)
            new_vals.append(v_full)

            if self.gqa > 1:
                k_expanded = k_full.repeat_interleave(self.gqa, dim=1)
                v_expanded = v_full.repeat_interleave(self.gqa, dim=1)
            else:
                k_expanded, v_expanded = k_full, v_full

            attn = F.scaled_dot_product_attention(
                q.float(),
                k_expanded.float(),
                v_expanded.float(),
                attn_mask=attn_mask,
                scale=1.0 / math.sqrt(self.hd),
            )
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.nh * self.hd)
            h = h + F.linear(attn, self.o_w[i])

            xn = self._rms(h, self.post_ln[i], self.eps)
            gate = F.silu(F.linear(xn, self.gate_w[i]))
            up = F.linear(xn, self.up_w[i])
            h = h + F.linear(gate * up, self.down_w[i])

        h = self._rms(h, self.final_norm, self.eps)
        logits = self.lm_head(h[:, -1:, :])

        return (logits, *new_keys, *new_vals)


def _check_saved(ov_dir, name):
    xml = ov_dir / f"{name}.xml"
    binn = ov_dir / f"{name}.bin"
    if xml.exists() and binn.exists():
        mb = binn.stat().st_size / 1e6
        print(f"  + {name}.xml + .bin ({mb:.1f} MB)")
    else:
        missing = [str(p) for p in (xml, binn) if not p.exists()]
        print(f"  x MISSING: {missing}")


def _make_decoder_stateful(decoder_ov, layer_count):
    try:
        from openvino._offline_transformations import apply_make_stateful_transformation
    except Exception as exc:
        raise RuntimeError(
            "Stateful decoder transform requires OpenVINO offline transformations support"
        ) from exc

    params = list(decoder_ov.get_parameters())
    results = list(decoder_ov.get_results())
    expected_params = 2 + 2 * layer_count
    expected_results = 1 + 2 * layer_count
    if len(params) < expected_params or len(results) < expected_results:
        raise RuntimeError(
            "Unexpected decoder IO layout for stateful transform: "
            f"params={len(params)} (expected >= {expected_params}), "
            f"results={len(results)} (expected >= {expected_results})"
        )

    state_pairs = list(zip(params[2 : 2 + 2 * layer_count], results[1 : 1 + 2 * layer_count]))
    apply_make_stateful_transformation(decoder_ov, state_pairs)
    return decoder_ov


def _resolve_compression_mode(nncf, weight_format, arg_name):
    if weight_format == "int8":
        return nncf.CompressWeightsMode.INT8_ASYM
    if weight_format == "int4":
        return nncf.CompressWeightsMode.INT4_ASYM
    raise ValueError(f"Unsupported {arg_name}: {weight_format}")


def convert(
    model_dir,
    ov_dir="ov_model",
    decoder_weight_format=None,
    thinker_weight_format=None,
):
    import openvino as ov

    ov_dir = Path(ov_dir)
    ov_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(model_dir)
    sf = MultiSafetensors(model_dir)
    nncf = None

    layer_count = cfg["dec_layers"]
    nkv = cfg["dec_kv_heads"]
    hd = cfg["dec_head_dim"]
    hidden = cfg["dec_hidden_size"]
    chunk_size = cfg["enc_n_window"] * 2

    # Copy exactly the tokenizer/preprocessor/model metadata files needed at runtime.
    for filename in EXTRA_MODEL_FILES:
        src = Path(model_dir) / filename
        dst = ov_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  ! missing optional file: {src}")

    print("=== Converting Audio Encoder ===")
    encoder = AudioEncoderModule(sf, cfg).eval()
    dummy_mel = torch.randn(1, 128, chunk_size * 16)

    with torch.no_grad():
        encoder_ov = ov.convert_model(
            encoder,
            example_input=dummy_mel,
            input=[ov.PartialShape([1, 128, -1])],
        )
    ov.save_model(encoder_ov, str(ov_dir / "audio_encoder_model.xml"))
    _check_saved(ov_dir, "audio_encoder_model")
    del encoder

    print("=== Converting Thinker Embeddings ===")
    thinker_embeddings = ThinkerEmbeddingModule(sf).eval()
    dummy_input_ids = torch.zeros(1, 16, dtype=torch.long)
    with torch.no_grad():
        thinker_embeddings_ov = ov.convert_model(
            thinker_embeddings,
            example_input=dummy_input_ids,
            input=[ov.PartialShape([1, -1])],
        )

    if thinker_weight_format is not None:
        try:
            nncf = importlib.import_module("nncf")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "NNCF is required for --thinker-weight-format. Install it with: pip install nncf"
            ) from exc

        thinker_compression_mode = _resolve_compression_mode(
            nncf,
            thinker_weight_format,
            "--thinker-weight-format",
        )
        print(
            f"=== Compressing Thinker Embeddings Weights to {thinker_weight_format.upper()} (NNCF) ==="
        )
        thinker_embeddings_ov = nncf.compress_weights(
            thinker_embeddings_ov,
            mode=thinker_compression_mode,
        )

    ov.save_model(thinker_embeddings_ov, str(ov_dir / "thinker_embeddings_model.xml"))
    _check_saved(ov_dir, "thinker_embeddings_model")
    del thinker_embeddings

    print("=== Converting Decoder ===")
    decoder = DecoderModule(sf, cfg).eval()
    seq_len, past_len = 8, 4
    dummy_embeds = torch.randn(1, seq_len, hidden)
    dummy_pos = torch.arange(past_len, past_len + seq_len).unsqueeze(0).long()
    dummy_pk = [torch.randn(1, nkv, past_len, hd) for _ in range(layer_count)]
    dummy_pv = [torch.randn(1, nkv, past_len, hd) for _ in range(layer_count)]
    example = (dummy_embeds, dummy_pos, *dummy_pk, *dummy_pv)

    input_shapes = [
        ov.PartialShape([1, -1, hidden]),
        ov.PartialShape([1, -1]),
    ]
    for _ in range(2 * layer_count):
        input_shapes.append(ov.PartialShape([1, nkv, -1, hd]))

    with torch.no_grad():
        decoder_ov = ov.convert_model(
            decoder,
            example_input=example,
            input=input_shapes,
        )

    print("=== Converting Decoder to Stateful KV-cache ===")
    decoder_ov = _make_decoder_stateful(decoder_ov, layer_count)

    if decoder_weight_format is not None:
        if nncf is None:
            try:
                nncf = importlib.import_module("nncf")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "NNCF is required for --decoder-weight-format. Install it with: pip install nncf"
                ) from exc

        compression_mode = _resolve_compression_mode(
            nncf,
            decoder_weight_format,
            "--decoder-weight-format",
        )

        print(
            f"=== Compressing Decoder Weights to {decoder_weight_format.upper()} (NNCF) ==="
        )
        decoder_ov = nncf.compress_weights(
            decoder_ov,
            mode=compression_mode,
        )

    ov.save_model(decoder_ov, str(ov_dir / "decoder_model.xml"))
    _check_saved(ov_dir, "decoder_model")
    del decoder

    print(f"\nConversion complete -> {ov_dir}/")
    print("  audio_encoder_model.xml/.bin")
    print("  thinker_embeddings_model.xml/.bin")
    print(f"  decoder_model.xml/.bin ({layer_count} layers, stateful prefill+decode)")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR OpenVINO converter")
    parser.add_argument("model_dir")
    parser.add_argument("--ov-dir", default="ov_model")
    parser.add_argument(
        "--decoder-weight-format",
        choices=("int8", "int4"),
        default=None,
        help="Apply NNCF decoder weight compression (int8 or int4)",
    )
    parser.add_argument(
        "--thinker-weight-format",
        choices=("int8", "int4"),
        default=None,
        help="Apply NNCF thinker embeddings weight compression (int8 or int4)",
    )
    args = parser.parse_args()
    convert(
        args.model_dir,
        args.ov_dir,
        decoder_weight_format=args.decoder_weight_format,
        thinker_weight_format=args.thinker_weight_format,
    )


if __name__ == "__main__":
    main()
