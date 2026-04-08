#!/usr/bin/env python3
"""
OpenVINO conversion script for Qwen3-ASR (0.6B / 1.7B).

Converts the PyTorch model into two OpenVINO IR sub-models:
  1. audio_encoder  – mel spectrogram [1, 128, T] → audio embeddings [1, N, output_dim]
  2. decoder        – unified prefill / decode with KV-cache

Usage:
    # Step 1 – Convert (creates ./ov_model/ with .xml/.bin pairs)
    pip install torch safetensors openvino numpy
    python convert_to_openvino.py convert <model_dir>

    # Step 2 – Inference
    pip install soundfile
    python convert_to_openvino.py infer <model_dir> <audio.wav> [--ov-dir ov_model]
"""

import sys, os, json, math, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ###########################################################################
#                           Config + helpers
# ###########################################################################

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


# ===================== Weight loading ======================================

from safetensors import safe_open


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


# ===================== Audio pre-processing ================================

SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400

TOKEN_IM_START = 151644
TOKEN_IM_END = 151645
TOKEN_AUDIO_START = 151669
TOKEN_AUDIO_END = 151670
TOKEN_AUDIO_PAD = 151676
TOKEN_ENDOFTEXT = 151643
TOKEN_ASR_TEXT = 151704
EOS_TOKEN_IDS = {TOKEN_ENDOFTEXT, TOKEN_IM_END}

PROMPT_PREFIX = [TOKEN_IM_START, 8948, 198, TOKEN_IM_END, 198,
                 TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START]
PROMPT_SUFFIX = [TOKEN_AUDIO_END, TOKEN_IM_END, 198,
                 TOKEN_IM_START, 77091, 198]


def hertz_to_mel(freq):
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= 1000.0
        mels[log_region] = 15.0 + np.log(freq[log_region] / 1000.0) * (27.0 / np.log(6.4))
    elif freq >= 1000.0:
        mels = 15.0 + np.log(freq / 1000.0) * (27.0 / np.log(6.4))
    return mels


def mel_to_hertz(mels):
    freq = 200.0 * mels / 3.0
    log_region = mels >= 15.0
    freq[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mels[log_region] - 15.0))
    return freq


def compute_mel_filters():
    num_freq = 1 + WINDOW_SIZE // 2
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, num_freq)
    mel_freqs = np.linspace(hertz_to_mel(0.0), hertz_to_mel(8000.0), NUM_MEL_BINS + 2)
    filter_freqs = mel_to_hertz(mel_freqs)
    fdiff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down = -slopes[:, :-2] / fdiff[:-1]
    up = slopes[:, 2:] / fdiff[1:]
    fb = np.maximum(0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2:NUM_MEL_BINS + 2] - filter_freqs[:NUM_MEL_BINS])
    fb *= enorm[np.newaxis, :]
    return fb.astype(np.float32)


def compute_mel_spectrogram(audio_np, mel_filters_np):
    audio = torch.from_numpy(audio_np).float()
    mel_filters = torch.from_numpy(mel_filters_np).float()
    window = torch.hann_window(WINDOW_SIZE)
    stft = torch.stft(audio, WINDOW_SIZE, HOP_LENGTH, window=window, return_complex=True)
    mag2 = stft[..., :-1].abs() ** 2
    mel_spec = mel_filters.T @ mag2
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy()


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


def count_encoder_tokens(total_frames, chunk_size=100):
    """Exact number of encoder output tokens for unpadded mel."""
    count = 0
    for start in range(0, total_frames, chunk_size):
        chunk_len = min(chunk_size, total_frames - start)
        t = chunk_len
        for _ in range(3):
            t = (t + 1) // 2
        count += t
    return count


# ###########################################################################
#            1. Audio Encoder  (batched chunks — no Python loops in graph)
# ###########################################################################

class AudioEncoderModule(nn.Module):
    """
    Input:  mel [1, 128, T]  – T MUST be pre-padded to a multiple of chunk_size
    Output: audio_embeds [1, total_tokens, output_dim]

    Chunking is done via reshape + batch so that the traced graph generalises to 
    any number of chunks.
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
        self.chunk_size = self.n_window * 2  # 100

        # Conv2D stem
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

        # Linear projection (no bias)
        self.conv_out = nn.Linear(dh * (NUM_MEL_BINS // 8), d, bias=False)
        self.conv_out.weight.data.copy_(_w(sf, f"{prefix}.conv_out.weight"))

        # Tokens-per-chunk is constant: 100 → 50 → 25 → 13
        tpc = self.chunk_size
        for _ in range(3):
            tpc = (tpc + 1) // 2
        self.tokens_per_chunk = tpc  # 13

        # Pre-compute sinusoidal position embedding (constant per chunk)
        self.register_buffer(
            "pos_emb",
            sinusoidal_position_embedding(self.tokens_per_chunk, d),
        )

        # Transformer layers
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

        # Post-LN
        self.ln_post_w = nn.Parameter(_w(sf, f"{prefix}.ln_post.weight"))
        self.ln_post_b = nn.Parameter(_w(sf, f"{prefix}.ln_post.bias"))

        # Projection head
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
        """mel: [1, 128, T] where T is pre-padded to a multiple of chunk_size."""
        cs = self.chunk_size

        # Reshape into batch of chunks: [nc, 1, 128, cs]
        mel_4d = mel.reshape(1, NUM_MEL_BINS, -1, cs)   # [1, 128, nc, cs]
        mel_4d = mel_4d.permute(2, 0, 1, 3)             # [nc, 1, 128, cs]
        mel_4d = mel_4d.reshape(-1, 1, NUM_MEL_BINS, cs) # [nc, 1, 128, cs]

        # Batched conv stem
        x = F.gelu(self.conv1(mel_4d))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        nc, c, f, tpc = x.shape
        x = x.permute(0, 3, 1, 2).reshape(nc, tpc, c * f)  # [nc, tpc, dh*freq]

        x = self.conv_out(x)  # [nc, tpc, d_model]

        # Per-chunk position embeddings (broadcast over all chunks)
        x = x + self.pos_emb.unsqueeze(0)

        # Flatten to single sequence: [1, N, d_model]
        x = x.reshape(1, nc * tpc, self.d_model)
        N = x.shape[1]

        # Windowed (block-diagonal) attention mask
        tpw = tpc * (self.n_window_infer // self.chunk_size)  # tokens per window (e.g. 104)
        idx = torch.arange(N, device=x.device)
        win_id = idx // tpw
        mask_bool = win_id.unsqueeze(0) != win_id.unsqueeze(1)  # [N, N]
        attn_mask = torch.where(
            mask_bool,
            torch.tensor(-1e9, device=x.device),
            torch.tensor(0.0, device=x.device),
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

        # Transformer layers
        nh, hd = self.n_heads, self.head_dim
        for i in range(self.n_layers):
            xn = F.layer_norm(x, (self.d_model,), self.attn_ln_w[i], self.attn_ln_b[i])
            q = F.linear(xn, self.q_proj_w[i], self.q_proj_b[i])
            k = F.linear(xn, self.k_proj_w[i], self.k_proj_b[i])
            v = F.linear(xn, self.v_proj_w[i], self.v_proj_b[i])

            q = q.view(1, N, nh, hd).transpose(1, 2)
            k = k.view(1, N, nh, hd).transpose(1, 2)
            v = v.view(1, N, nh, hd).transpose(1, 2)

            attn = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                attn_mask=attn_mask, scale=1.0 / math.sqrt(hd),
            )
            attn = attn.transpose(1, 2).contiguous().view(1, N, nh * hd)
            x = x + F.linear(attn, self.out_proj_w[i], self.out_proj_b[i])

            xn = F.layer_norm(x, (self.d_model,), self.ffn_ln_w[i], self.ffn_ln_b[i])
            x = x + F.linear(F.gelu(F.linear(xn, self.fc1_w[i], self.fc1_b[i])),
                              self.fc2_w[i], self.fc2_b[i])

        # Post-LN + projection
        x = F.layer_norm(x, (self.d_model,), self.ln_post_w, self.ln_post_b)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x  # [1, N, output_dim]


# ###########################################################################
#  2. Unified Decoder  (prefill + decode, single set of weights)
# ###########################################################################

class DecoderModule(nn.Module):
    """
    Inputs:
        input_embeds  [1, S, H]     – S=prompt_len for prefill, S=1 for decode
        position_ids  [1, S]        – int64, absolute positions
        past_k_i      [1, nkv, P, hd]  × L   (P=0 for initial prefill)
        past_v_i      [1, nkv, P, hd]  × L
    Outputs:
        logits        [1, 1, V]     – last-token logits only
        new_k_i       [1, nkv, P+S, hd]  × L
        new_v_i       [1, nkv, P+S, hd]  × L
    """

    def __init__(self, sf, cfg):
        super().__init__()
        self.H = cfg["dec_hidden_size"]
        self.L = cfg["dec_layers"]
        self.nh = cfg["dec_heads"]
        self.nkv = cfg["dec_kv_heads"]
        self.hd = cfg["dec_head_dim"]
        self.eps = cfg["dec_rms_norm_eps"]
        self.theta = cfg["dec_rope_theta"]
        self.V = cfg["dec_vocab_size"]
        self.gqa = self.nh // self.nkv

        self.lm_head = nn.Linear(self.H, self.V, bias=False)
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

        for i in range(self.L):
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
        """
        past_kv layout: k0, k1, ..., k_{L-1}, v0, v1, ..., v_{L-1}
        each [1, nkv, past_len, hd].  past_len=0 on first prefill call.
        """
        B, S, _ = input_embeds.shape
        L = self.L
        past_keys = list(past_kv[:L])
        past_vals = list(past_kv[L:])
        past_len = past_keys[0].shape[2]
        total_kv = past_len + S

        # RoPE
        positions = position_ids.squeeze(0)  # [S]
        rope_cos, rope_sin = compute_rope_freqs(positions, self.hd, self.theta)
        rc = rope_cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, hd]
        rs = rope_sin.unsqueeze(0).unsqueeze(2)
        half = self.hd // 2

        # Causal mask: query at position p attends to kv positions ≤ p
        q_pos = position_ids.float()                                         # [1, S]
        kv_pos = torch.arange(total_kv, device=input_embeds.device).float()  # [total_kv]
        mask_bool = kv_pos.unsqueeze(0) > q_pos.unsqueeze(-1)                # [1, S, total_kv]
        attn_mask = torch.where(
            mask_bool,
            torch.tensor(-1e9, device=input_embeds.device),
            torch.tensor(0.0, device=input_embeds.device),
        ).unsqueeze(1)  # [1, 1, S, total_kv]

        h = input_embeds
        new_keys = []
        new_vals = []

        for i in range(L):
            xn = self._rms(h, self.in_ln[i], self.eps)

            q = F.linear(xn, self.q_w[i]).view(B, S, self.nh, self.hd)
            k = F.linear(xn, self.k_w[i]).view(B, S, self.nkv, self.hd)
            v = F.linear(xn, self.v_w[i]).view(B, S, self.nkv, self.hd)

            # Per-head Q/K RMSNorm
            q = self._rms(q, self.q_norm[i], self.eps)
            k = self._rms(k, self.k_norm[i], self.eps)

            # RoPE
            q = q * rc + torch.cat([-q[..., half:], q[..., :half]], -1) * rs
            k = k * rc + torch.cat([-k[..., half:], k[..., :half]], -1) * rs

            q = q.transpose(1, 2)  # [B, nh, S, hd]
            k = k.transpose(1, 2)  # [B, nkv, S, hd]
            v = v.transpose(1, 2)

            # Concatenate with past
            k_full = torch.cat([past_keys[i], k], dim=2)
            v_full = torch.cat([past_vals[i], v], dim=2)
            new_keys.append(k_full)
            new_vals.append(v_full)

            # GQA expand
            if self.gqa > 1:
                ke = k_full.repeat_interleave(self.gqa, dim=1)
                ve = v_full.repeat_interleave(self.gqa, dim=1)
            else:
                ke, ve = k_full, v_full

            attn = F.scaled_dot_product_attention(
                q.float(), ke.float(), ve.float(),
                attn_mask=attn_mask, scale=1.0 / math.sqrt(self.hd),
            )
            attn = attn.transpose(1, 2).contiguous().view(B, S, self.nh * self.hd)
            h = h + F.linear(attn, self.o_w[i])

            # SwiGLU MLP
            xn = self._rms(h, self.post_ln[i], self.eps)
            gate = F.silu(F.linear(xn, self.gate_w[i]))
            up = F.linear(xn, self.up_w[i])
            h = h + F.linear(gate * up, self.down_w[i])

        h = self._rms(h, self.final_norm, self.eps)
        logits = self.lm_head(h[:, -1:, :])  # [B, 1, V]

        return (logits, *new_keys, *new_vals)


# ###########################################################################
#                        Conversion
# ###########################################################################

def convert(model_dir, ov_dir="ov_model"):
    import openvino as ov

    ov_dir = Path(ov_dir)
    ov_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(model_dir)
    sf = MultiSafetensors(model_dir)

    L = cfg["dec_layers"]
    nkv = cfg["dec_kv_heads"]
    hd = cfg["dec_head_dim"]
    H = cfg["dec_hidden_size"]
    cs = cfg["enc_n_window"] * 2  # chunk_size

    with open(ov_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    core = ov.Core()

    # ---- 1. Audio Encoder ------------------------------------------------
    print("=== Converting Audio Encoder ===")
    encoder = AudioEncoderModule(sf, cfg).eval()
    dummy_mel = torch.randn(1, 128, cs * 16)  # 16 chunks, must be multiple of cs

    with torch.no_grad():
        encoder_ov = ov.convert_model(
            encoder,
            example_input=dummy_mel,
            input=[ov.PartialShape([1, 128, -1])],
        )
    ov.save_model(encoder_ov, str(ov_dir / "audio_encoder.xml"))
    _check_saved(ov_dir, "audio_encoder")
    del encoder

    # ---- 2. Unified Decoder ----------------------------------------------
    print("=== Converting Decoder ===")
    decoder = DecoderModule(sf, cfg).eval()

    # Trace with S=8, past_len=4  (both > 1 for good generalisation)
    S_d, P_d = 8, 4
    dummy_embeds = torch.randn(1, S_d, H)
    dummy_pos = torch.arange(P_d, P_d + S_d).unsqueeze(0).long()
    dummy_pk = [torch.randn(1, nkv, P_d, hd) for _ in range(L)]
    dummy_pv = [torch.randn(1, nkv, P_d, hd) for _ in range(L)]
    example = (dummy_embeds, dummy_pos, *dummy_pk, *dummy_pv)

    input_shapes = [
        ov.PartialShape([1, -1, H]),
        ov.PartialShape([1, -1]),
    ]
    for _ in range(2 * L):
        input_shapes.append(ov.PartialShape([1, nkv, -1, hd]))

    with torch.no_grad():
        decoder_ov = ov.convert_model(
            decoder,
            example_input=example,
            input=input_shapes,
        )
    ov.save_model(decoder_ov, str(ov_dir / "decoder.xml"))
    _check_saved(ov_dir, "decoder")
    del decoder

    print(f"\n✓  Conversion complete → {ov_dir}/")
    print(f"   audio_encoder.xml/.bin")
    print(f"   decoder.xml/.bin  ({L} layers, unified prefill+decode)")


def _check_saved(ov_dir, name):
    xml = ov_dir / f"{name}.xml"
    binn = ov_dir / f"{name}.bin"
    if xml.exists() and binn.exists():
        mb = binn.stat().st_size / 1e6
        print(f"  ✓ {name}.xml + .bin ({mb:.1f} MB)")
    else:
        missing = [str(p) for p in (xml, binn) if not p.exists()]
        print(f"  ✗ MISSING: {missing}")


# ###########################################################################
#                        Inference
# ###########################################################################

def infer(model_dir, wav_path, ov_dir="ov_model", device="CPU"):
    import openvino as ov
    import soundfile as sf_audio

    ov_dir = Path(ov_dir)
    with open(ov_dir / "config.json") as f:
        cfg = json.load(f)

    core = ov.Core()
    print("Loading OpenVINO models...", file=sys.stderr)
    enc_model = core.compile_model(str(ov_dir / "audio_encoder.xml"), device)
    dec_model = core.compile_model(str(ov_dir / "decoder.xml"), device)

    L = cfg["dec_layers"]
    nkv = cfg["dec_kv_heads"]
    hd = cfg["dec_head_dim"]
    H = cfg["dec_hidden_size"]
    cs = cfg["enc_n_window"] * 2

    # Embedding table
    sf_weights = MultiSafetensors(model_dir)
    tok_emb_np = _w(sf_weights, "thinker.model.embed_tokens.weight").numpy()

    # --- Audio ---
    audio_array, sr = sf_audio.read(wav_path, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        new_len = int(len(audio_array) * ratio)
        indices = np.linspace(0, len(audio_array) - 1, new_len)
        audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.float32)

    mel_filters = compute_mel_filters()
    mel = compute_mel_spectrogram(audio_array, mel_filters)  # [128, T]
    total_frames = mel.shape[1]
    expected_tokens = count_encoder_tokens(total_frames, cs)
    print(f"Mel: {mel.shape}, expect {expected_tokens} encoder tokens", file=sys.stderr)

    # Pad to multiple of chunk_size
    pad_len = (cs - total_frames % cs) % cs
    if pad_len > 0:
        mel = np.pad(mel, ((0, 0), (0, pad_len)))
    mel_input = mel[np.newaxis, :, :].astype(np.float32)

    # --- Encoder ---
    print("Running encoder...", file=sys.stderr)
    enc_out = enc_model([mel_input])
    audio_embeds = enc_out[enc_model.output(0)]            # [1, N_padded, dim]
    audio_embeds = audio_embeds[0, :expected_tokens, :]    # [N, dim]  trim padding tokens
    n_audio = audio_embeds.shape[0]
    print(f"Audio embeddings: {audio_embeds.shape}", file=sys.stderr)

    # --- Build prompt ---
    input_ids = PROMPT_PREFIX + [TOKEN_AUDIO_PAD] * n_audio + PROMPT_SUFFIX
    input_embeds = tok_emb_np[input_ids].copy()  # [S, H]
    pad_start = len(PROMPT_PREFIX)
    input_embeds[pad_start:pad_start + n_audio] = audio_embeds
    input_embeds = input_embeds[np.newaxis, :, :].astype(np.float32)

    S = len(input_ids)
    position_ids = np.arange(S, dtype=np.int64)[np.newaxis, :]

    # --- Prefill (empty past, past_len = 0) ---
    print(f"Running prefill ({S} tokens)...", file=sys.stderr)
    empty_past = np.zeros((1, nkv, 0, hd), dtype=np.float32)
    dec_inputs = [input_embeds, position_ids] + [empty_past] * (2 * L)
    dec_out = dec_model(dec_inputs)

    logits = dec_out[dec_model.output(0)]
    kv_caches = [dec_out[dec_model.output(1 + i)] for i in range(2 * L)]

    token = int(np.argmax(logits[0, 0]))
    generated = [token]

    # --- Autoregressive decode ---
    print("Generating...", file=sys.stderr)
    max_new_tokens = 1024
    for step in range(max_new_tokens - 1):
        if token in EOS_TOKEN_IDS:
            break
        pos = S + step
        embed = tok_emb_np[token][np.newaxis, np.newaxis, :].astype(np.float32)
        pos_id = np.array([[pos]], dtype=np.int64)

        dec_out = dec_model([embed, pos_id] + kv_caches)
        logits = dec_out[dec_model.output(0)]
        kv_caches = [dec_out[dec_model.output(1 + i)] for i in range(2 * L)]

        token = int(np.argmax(logits[0, 0]))
        generated.append(token)

    while generated and generated[-1] in EOS_TOKEN_IDS:
        generated.pop()

    print(f"Generated {len(generated)} tokens", file=sys.stderr)
    text = decode_tokens(generated, model_dir)
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text


# ###########################################################################
#                        Tokenizer
# ###########################################################################

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def decode_tokens(token_ids, model_dir):
    vocab_path = os.path.join(model_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    special_tokens = set()
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        for tid_str in tc.get("added_tokens_decoder", {}):
            special_tokens.add(int(tid_str))

    byte_enc = bytes_to_unicode()
    byte_dec = {v: k for k, v in byte_enc.items()}

    pieces = []
    for tid in token_ids:
        if tid in special_tokens:
            if tid == TOKEN_ASR_TEXT:
                pieces.append("<asr_text>")
            continue
        tok = id_to_token.get(tid, "")
        if tok:
            pieces.append(tok)
    text = "".join(pieces)
    return bytearray([byte_dec[c] for c in text if c in byte_dec]).decode("utf-8", errors="replace")


# ###########################################################################
#                                CLI
# ###########################################################################

def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR → OpenVINO converter")
    sub = parser.add_subparsers(dest="cmd")

    p_conv = sub.add_parser("convert", help="Convert PyTorch → OpenVINO IR")
    p_conv.add_argument("model_dir")
    p_conv.add_argument("--ov-dir", default="ov_model")

    p_inf = sub.add_parser("infer", help="Run inference with OpenVINO")
    p_inf.add_argument("model_dir", help="Original model dir (for tokenizer + embeddings)")
    p_inf.add_argument("wav_path")
    p_inf.add_argument("--ov-dir", default="ov_model")
    p_inf.add_argument("--device", default="CPU")

    args = parser.parse_args()
    if args.cmd == "convert":
        convert(args.model_dir, args.ov_dir)
    elif args.cmd == "infer":
        text = infer(args.model_dir, args.wav_path, args.ov_dir, args.device)
        print(text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
