# Install

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13) — for the default RTX 30/40/50 path
- For **Pascal (sm_61, GTX 10-series)** GPUs: driver with CUDA 12.6 support and a
  CUDA 12.6 toolkit (nvcc on PATH). See [Pascal (sm_61)](#pascal-sm_61-gtx-10-series)
  below — the default CUDA 13 wheel dropped Pascal.
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

## Method 2: Install from source

```bash
git clone https://github.com/Blue-nim/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Pascal (sm_61, GTX 10-series)

PyTorch 2.11+ ships CUDA 13 wheels that **dropped Maxwell/Pascal** (CUDA 13 supports
only Turing 7.5+). A GTX 1070/1080 will error with `no kernel image is available for
execution on the device` under the default install. To run on Pascal:

```bash
git clone https://github.com/Blue-nim/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
FREETOKEN_PASCAL=1 uv pip install -e .   # or use install.sh with FREETOKEN_PASCAL=1
```

`FREETOKEN_PASCAL=1` selects the **CUDA 12.6 (cu126)** torch wheel (last line with
Pascal SASS), installs the **base** package (the cu13-only `accel` extras — flashinfer
`cu13`, sglang-kernel — cannot install against cu126), and builds the kernel-cache with
an `sm_61` cubin. The GGUF dequant/matmul kernels are Pascal-safe: they use DP4A int8 dot
products and fp32 scale math (no native fp16 arithmetic, which consumer Pascal lacks), so
they compile and run correctly — just slower than on Ampere+. bf16 output degrades to a
float conversion.

> Note: no single torch wheel covers both Pascal (needs cu126) and Blackwell/RTX 50
> (needs cu130). The default, non-Pascal install targets RTX 30/40/50.

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
