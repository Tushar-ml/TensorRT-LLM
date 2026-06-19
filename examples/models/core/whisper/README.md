# Whisper

> [!WARNING]
> The `convert_checkpoint.py` / `trtllm-build` / `run.py` workflow described
> below is **legacy** and will not receive new features. New projects should use
> [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
> or the [LLM Python API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html) instead.

This document shows how to build and run a [whisper model](https://github.com/openai/whisper/tree/main) in TensorRT LLM on a single GPU.

- [Whisper](#whisper)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
    - [Run](#run)
      - [Run C++ runtime](#run-c-runtime)
      - [Run Python runtime](#run-python-runtime)
      - [Advanced Usage](#advanced-usage)
    - [Draft-Target Speculative Decoding](#draft-target-speculative-decoding)
    - [Distil-Whisper](#distil-whisper)
    - [Acknowledgment](#acknowledgment)

## Overview

The TensorRT LLM Whisper example code is located in [`examples/models/core/whisper`](./).

 * [`convert_checkpoint.py`](./convert_checkpoint.py) to convert weights from OpenAI Whisper format to TRT-LLM format.
 * `trtllm-build` to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Whisper model.
 * [`run.py`](./run.py) to run the inference on a single wav file, or [a HuggingFace dataset](https://huggingface.co/datasets/openslr/librispeech_asr) [\(Librispeech test clean\)](https://www.openslr.org/12).

## Support Matrix
  * FP16
  * INT8 (Weight Only Quant)
  * INT4 (Weight Only Quant)

## Usage

The TensorRT LLM Whisper example code locates at [examples/models/core/whisper](./). It takes whisper pytorch weights as input, and builds the corresponding TensorRT engines.

### Build TensorRT engine(s)

Need to prepare the whisper checkpoint first by downloading models from [here](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L22-L28).


```bash
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
# take large-v3 model as an example
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
```

TensorRT LLM Whisper builds TensorRT engine(s) from the pytorch checkpoint.

```bash
# install requirements first
pip install -r requirements.txt

INFERENCE_PRECISION=float16
WEIGHT_ONLY_PRECISION=int8
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
checkpoint_dir=whisper_large_v3_weights_${WEIGHT_ONLY_PRECISION}
output_dir=whisper_large_v3_${WEIGHT_ONLY_PRECISION}

# Convert the large-v3 model weights into TensorRT LLM format.
python3 convert_checkpoint.py \
                --use_weight_only \
                --weight_only_precision $WEIGHT_ONLY_PRECISION \
                --output_dir $checkpoint_dir

# Build the large-v3 model using trtllm-build
trtllm-build  --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --moe_plugin disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --max_input_len 3000 --max_seq_len=3000

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}
```

### Run
Different types of runtime are provided for whisper models. Following an order of serving performance and good usability, we recommend:
- (NEW) Python binding of C++ runtime w/ Paged KV Cache and Inflight Batching (IFB)
- Python runtime w/ Static Batching

Please refer to the documentation for the details of [paged kv cache](../../../../docs/source/legacy/advanced/gpt-attention.md#paged-kv-cache) and [inflight batching](../../../../docs/source/legacy/advanced/gpt-attention.md#inflight-batching).

#### Run C++ runtime
**Note: to use inflight batching and paged kv cache features in C++ runtime, please make sure you have set `--paged_kv_cache enable` and `--remove_input_padding enable` (which is by default enabled) in the `trtllm-build` command. Meanwhile, if using Python runtime, it is recommended to disable these flag by `--paged_kv_cache disable` and `--remove_input_padding disable` to avoid any unnecessary overhead.**

```bash
# choose the engine you build [./whisper_large_v3, ./whisper_large_v3_int8]
output_dir=./whisper_large_v3
# decode a single audio file
# If the input file does not have a .wav extension, ffmpeg needs to be installed with the following command:
# apt-get update && apt-get install -y ffmpeg
# Inferencing via python binding of C++ runtime with inflight batching (IFB)
python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/1221-135766-0002.wav
# decode a whole dataset
python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3
```


For pure C++ runtime, there is no example given yet. Please check the [`Executor`](../../../../cpp/include/tensorrt_llm/executor/executor.h) API to implement your own end-to-end workflow. It is highly recommended to leverage more encapsulated solutions such as the above C++ Python binding or [Triton backend](https://github.com/triton-inference-server/tensorrtllm_backend).

<!-- #### Run with Triton Backend
[Triton backend](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/whisper.md) contains the tutorial on how to run whisper engines with Tritonserver. -->

#### Run Python runtime

For pure Python runtime, you can simply add the `--use_py_session` option.

#### Advanced Usage

`--padding_strategy`
OpenAI's official Whisper models accept WAV files of up to 30 seconds in length. For files shorter than 30 seconds, padding is required to reach the 30-second mark, which may not be efficient. Currently, three padding strategies are supported:

1. **max (default)**: Pads to 30 seconds.
2. **longest**: Pads according to the longest duration in the current batch.
3. **nopad**: No padding is applied. You will need to fine-tune the Whisper model to maintain accuracy. See [examples](https://github.com/k2-fsa/icefall/blob/master/egs/aishell/ASR/whisper/whisper_encoder_forward_monkey_patch.py#L15).

`--text_prefix`
You can modify the input prompt for the Whisper decoder. For example, use `<|startoftranscript|><|en|><|zh|><|transcribe|><|notimestamps|>` to perform code-switching ASR between Chinese and English.

`--compute_cer`
Calculates the character error rate (CER) instead of the word error rate (WER) for languages such as Chinese and Japanese.

`--dataset`, `--dataset_name`, and `--dataset_split`
These options allow you to select different decoding audio datasets from Hugging Face.

### Draft-Target Speculative Decoding

Accelerate greedy Whisper large-v3 decoding with a smaller draft decoder (e.g. `large-v3-turbo`) using [draft-target speculative decoding](../../../../docs/source/legacy/advanced/speculative-decoding.md). v1 supports **greedy decoding only** (`num_beams=1`).

Build four engine dirs: draft (turbo), plain large-v3 baseline (quality reference), and DTM target (large-v3 encoder + spec-dec decoder). Turbo and large-v3 share the tokenizer/vocab but **DTM transcript quality follows the large-v3 target**, not the turbo draft.

```bash
MAX_DRAFT_LEN=4
DRAFT_CKPT=whisper_turbo_weights
TARGET_CKPT=whisper_large_v3_weights
DRAFT_ENGINE=whisper_turbo_engine
BASELINE_ENGINE=whisper_large_v3_engine
TARGET_ENGINE=whisper_large_v3_dtm_engine

# Draft model (turbo) — standard decoder build
python3 convert_checkpoint.py --model_name large-v3-turbo --output_dir ${DRAFT_CKPT}
# Encoder: enable padding removal for C++ IFB runtime; explicitly disable paged KV
trtllm-build --checkpoint_dir ${DRAFT_CKPT}/encoder --output_dir ${DRAFT_ENGINE}/encoder \
  --moe_plugin disable --max_batch_size 8 --gemm_plugin disable \
  --bert_attention_plugin float16 --max_input_len 3000 --max_seq_len 3000 \
  --remove_input_padding enable --kv_cache_type disabled
# Decoder: C++ IFB runtime (paged KV + padding removal)
trtllm-build --checkpoint_dir ${DRAFT_CKPT}/decoder --output_dir ${DRAFT_ENGINE}/decoder \
  --moe_plugin disable --max_batch_size 8 --max_beam_width 1 \
  --max_seq_len 114 --max_input_len 14 --max_encoder_input_len 3000 \
  --gemm_plugin float16 --bert_attention_plugin float16 --gpt_attention_plugin float16 \
  --remove_input_padding enable --kv_cache_type paged --use_paged_context_fmha enable

# Target model (large-v3) — plain baseline + spec-dec DTM target (same checkpoint)
python3 convert_checkpoint.py --model_name large-v3 --model_dir assets --output_dir ${TARGET_CKPT}

# Plain large-v3 baseline (validate weights / compare quality before DTM)
trtllm-build --checkpoint_dir ${TARGET_CKPT}/encoder --output_dir ${BASELINE_ENGINE}/encoder \
  --moe_plugin disable --max_batch_size 8 --gemm_plugin disable \
  --bert_attention_plugin float16 --max_input_len 3000 --max_seq_len 3000 \
  --remove_input_padding enable --kv_cache_type disabled
trtllm-build --checkpoint_dir ${TARGET_CKPT}/decoder --output_dir ${BASELINE_ENGINE}/decoder \
  --moe_plugin disable --max_batch_size 8 --max_beam_width 1 \
  --max_seq_len 114 --max_input_len 14 --max_encoder_input_len 3000 \
  --gemm_plugin float16 --bert_attention_plugin float16 --gpt_attention_plugin float16 \
  --remove_input_padding enable --kv_cache_type paged --use_paged_context_fmha enable

# DTM target (rebuild encoder + spec-dec decoder; do not reuse a broken encoder build)
trtllm-build --checkpoint_dir ${TARGET_CKPT}/encoder --output_dir ${TARGET_ENGINE}/encoder \
  --moe_plugin disable --max_batch_size 8 --gemm_plugin disable \
  --bert_attention_plugin float16 --max_input_len 3000 --max_seq_len 3000 \
  --remove_input_padding enable --kv_cache_type disabled
trtllm-build --checkpoint_dir ${TARGET_CKPT}/decoder --output_dir ${TARGET_ENGINE}/decoder \
  --moe_plugin disable --max_batch_size 8 --max_beam_width 1 \
  --max_seq_len 114 --max_input_len 14 --max_encoder_input_len 3000 \
  --speculative_decoding_mode draft_tokens_external --max_draft_len ${MAX_DRAFT_LEN:-16} \
  --gemm_plugin float16 --bert_attention_plugin float16 --gpt_attention_plugin float16 \
  --remove_input_padding enable --kv_cache_type paged --use_paged_context_fmha enable
```

Validate large-v3 baseline before DTM (compare to turbo separately for speed only):

```bash
python3 run.py --engine_dir ${BASELINE_ENGINE} --input_file ${AUDIO} \
  --assets_dir assets --num_beams 1 --use_py_session
```

Run optimized DTM inference (single GPU; build target decoder with `--max_draft_len 16`):

```bash
python3 run_dtm.py \
  --draft_engine_dir ${DRAFT_ENGINE} \
  --target_engine_dir ${TARGET_ENGINE} \
  --baseline_engine_dir ${BASELINE_ENGINE} \
  --draft_target_model_config="[16,[0],[0],False]" \
  --draft_mode turbo \
  --input_file /home/ubuntu/audio_60s_30s.wav \
  --assets_dir assets \
  --kv_cache_enable_block_reuse \
  --draft_kv_cache_free_gpu_memory_fraction 0.18 \
  --target_kv_cache_free_gpu_memory_fraction 0.28 \
  --profile
```

Benchmark turbo vs large-v3 vs optimized DTM (single H100, 30s audio; DTM target decoder built with `--max_draft_len 16`):

| Mode | Wall time | RTF |
|------|-----------|-----|
| Turbo | ~0.075s | 0.0025 |
| Large-v3 | ~0.35s | 0.012 |
| DTM (`draft_len=16`) | **~0.21s** | **0.007** |

```bash
python3 benchmark_dtm.py \
  --input_file /home/ubuntu/audio_60s_30s.wav \
  --assets_dir assets \
  --draft_engine_dir ${DRAFT_ENGINE} \
  --target_engine_dir ${TARGET_ENGINE} \
  --baseline_engine_dir ${BASELINE_ENGINE}
```

Optional draft engine rebuild for tighter single-batch KV pools:

```bash
MAX_BATCH=1 bash build_draft_engines.sh
# distil draft experiment:
# DRAFT_MODEL=distil-large-v3 DRAFT_ENGINE=whisper_distil_draft_engine bash build_draft_engines.sh
```

Legacy DTM invocation (full ModelRunnerCpp draft path):

```bash
python3 run_dtm.py \
  --draft_engine_dir ${DRAFT_ENGINE} \
  --target_engine_dir ${TARGET_ENGINE} \
  --draft_target_model_config="[4,[0],[1],False]" \
  --input_file assets/1221-135766-0002.wav \
  --assets_dir assets \
  --kv_cache_enable_block_reuse \
  --kv_cache_free_gpu_memory_fraction 0.45
```

Notes:
- `--draft_target_model_config` is `[draft_len, draft_gpu_ids, target_gpu_ids, use_logits]`.
- `--draft_len` overrides draft length (default **16** from config; rebuild DTM decoder with `--max_draft_len` ≥ chosen value).
- `--sweep_draft_len` benchmarks `{4,6,8,10,12,16}`.
- `--draft_mode` supports `turbo`, `ngram`, or `hybrid` draft proposals.
- `--draft_backend cpp` (default) keeps ModelRunnerCpp draft quality; `py` uses encoder-once py_session draft.
- `--profile` prints per-iteration draft/target latency.
- DTM output should match `${BASELINE_ENGINE}` greedy decoding; turbo (`${DRAFT_ENGINE}`) is a different model and may differ in wording.
- Encoder builds must use `--remove_input_padding enable --kv_cache_type disabled` (TRT-LLM 1.3+ defaults break Whisper encoder otherwise).
- Use leader (non-orchestrator) mode for enc-dec engines; orchestrator workers do not load the Whisper encoder.
- KV cache block reuse must be enabled on the target decoder engine.
- Beam search is not supported with speculative decoding.
- Optional: pass `--baseline_engine_dir ${BASELINE_ENGINE}` to `run_dtm.py` to print the plain large-v3 transcript before DTM.

### Distil-Whisper
TensorRT LLM also supports using [distil-whisper's](https://github.com/huggingface/distil-whisper) different models by first converting their params and weights from huggingface's naming format to [openai whisper](https://github.com/openai/whisper) naming format.
You can do so by running the script [distil_whisper/convert_from_distil_whisper.py](./convert_from_distil_whisper.py) as follows:

```bash
# take distil-medium.en as an example
# download the gpt2.tiktoken
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken

# will download the model weights from huggingface and convert them to openai-whisper's pytorch format
# model is saved to ./assets/ by default
python3 distil_whisper/convert_from_distil_whisper.py --model_name distil-whisper/distil-medium.en --output_name distil-medium.en

# convert whisper model weights into TensorRT-LLM
INFERENCE_PRECISION=float16
WEIGHT_ONLY_PRECISION=int8
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
checkpoint_dir=distil_whisper_medium_en_weights_${WEIGHT_ONLY_PRECISION}
output_dir=distil_whisper_medium_en${WEIGHT_ONLY_PRECISION}

python3 convert_checkpoint.py \
                --use_weight_only \
                --weight_only_precision $WEIGHT_ONLY_PRECISION \
                --output_dir $checkpoint_dir \
                --model_name distil-medium.en
```

<details><summary> Now, we can build and run the model like before: </summary><p>

```

trtllm-build  --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --moe_plugin disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --max_input_len 3000 --max_seq_len=3000

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}

# use cpp runtime python bindings
python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --name librispeech_dummy_${output_dir}
# use python runtime
python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --name librispeech_dummy_${output_dir} --use_py_session
```
</details>

### Acknowledgment

This implementation of TensorRT LLM for Whisper has been adapted from the [NVIDIA TensorRT LLM Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) submission of Jinheng Wang, which can be found in the repository [Eddie-Wang-Hackathon2023](https://github.com/Eddie-Wang1120/Eddie-Wang-Hackathon2023) on GitHub. We extend our gratitude to Jinheng for providing a foundation for the implementation.
