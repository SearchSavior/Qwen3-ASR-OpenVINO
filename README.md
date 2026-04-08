
## Qwen3-ASR OpenVINO

This repository contains an OpenVINO implementation of Qwen3-ASR that I built on top of [antirez/qwen-asr](https://github.com/antirez/qwen-asr/blob/main/python_simple_implementation.py), which I found by accident while poking through Deepwiki in February 2026. `qwen3_asr_utils.py`was adapted from [utils.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py) in the offical Qwen repo. This implemention is was signifigantly upgraded and is now deployed in [OpenArc](https://github.com/SearchSavior/OpenArc).

Originally this code was optimized for execution with Xeon W-2255 and Intel Arc A770.

The code in OpenArc was signifigantly upgrades for the serving usecase, but this repo is meant to finally provide somewhat approachable clarity to how exactly OpenVINO model optimization works, without library complexity. 

Claude Code with Opus 4.6 was used to develop this code, which was an adventure. Opus did not make very good openvino flavored design choices, mostly because pretrain data on openvio repos is fresh, and without many quality examples. Basically it took a ton of steering and my experience with openvino to make good choices. 

## Original Notebook

Most of the really OpenVINO lore is obfuscated by horrendous documentation around the engineering practices in the actual repositories. So, a goal of this project is to spell out exactly what it looks like to throw away `transformers`, and reimplement an openvino IR from scratch.

I drew  inspiration from the OpenVINO Notebooks approach to subgraph structure and stateful execution. Only Qwen3-ASR-0.6B has been validated. There were many other design decisions along the way, and I wish I had taken better notes.

Usually in the openvino/notebooks repo they purge code once the library adds support, so to maintain continutity the notebook files I d live in this repo. 

Original notebook is [here](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen3-asr), but will look very different in the future.



CPU and GPU devices are covered by the IR this repository can generate; NPU device would require some changes I don't have a device to evaluate. Such modifications are a claudable problem, and most coding agents these days should be able to handle. Most of the work is deciding what devices to use for subgraphs, and in NPU case, avoiding dynamic shape — or applying them when it's clever to do so.




---

## Install

Python 3.11 is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install safetensors openvino nncf numpy soundfile
```

---

## Convert

Download the Qwen3-ASR model weights from Hugging Face (e.g. `Qwen/Qwen3-ASR-0.6B`) into a local directory, then run:

```bash
python qwen3_asr_ov_convert.py <model_dir> [--ov-dir ov_model]
```

Optional weight compression via NNCF:

```bash
# INT8 decoder compression
python qwen3_asr_ov_convert.py <model_dir> --ov-dir ov_model_int8 --decoder-weight-format int8

# INT4 decoder compression
python qwen3_asr_ov_convert.py <model_dir> --ov-dir ov_model_int4 --decoder-weight-format int4

# INT8 thinker embeddings compression
python qwen3_asr_ov_convert.py <model_dir> --ov-dir ov_model_int8 --thinker-weight-format int8
```

The converter produces three IR sub-models inside the output directory:

```
ov_model/
  audio_encoder_model.xml / .bin
  thinker_embeddings_model.xml / .bin
  decoder_model.xml / .bin
```

---

## Infer

```bash
python qwen3_asr_ov_infer.py <audio.wav> [--ov-dir ov_model] [--device CPU]
```

Transcribed text is printed to stdout; performance metrics are printed to stderr.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `wav_path` | *(required)* | Path to a WAV audio file |
| `--ov-dir` | `ov_model` | Directory containing the OpenVINO IR models |
| `--device` | `CPU` | OpenVINO device string (`CPU`, `GPU`, `GPU.0`, etc.) |
| `--language` | auto-detect | Force a language (e.g. `English`, `Chinese`) |
| `--max-tokens` | `1024` | Maximum generated tokens per audio chunk |
| `--max-chunk-sec` | `30.0` | Target audio chunk size in seconds |
| `--search-expand-sec` | `5.0` | Boundary search window in seconds around each chunk cut |
| `--min-window-ms` | `100.0` | Sliding energy window in ms for silence detection |

### Examples

```bash
# Basic transcription on CPU
python qwen3_asr_ov_infer.py audio.wav

# GPU inference with a specific model directory
python qwen3_asr_ov_infer.py audio.wav --ov-dir ov_model_int8 --device GPU

# Force English with increased token budget
python qwen3_asr_ov_infer.py audio.wav --language English --max-tokens 2048
```
