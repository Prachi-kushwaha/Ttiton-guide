# Triton:GPU Programming A to Z?

## What is Triton
Triton is open-source programming language and compiler developed by OPENAI. it lets you write high-performance CUDA kernels in python. triton sits betweeen CUDA(too low-level) and pytorch(too high-level)

## Why not raw CUDA?
CUDA requires managing thread blocks, warps shared memory layouts, bank conflicts, and register pressure manually. Triton automates these while letting you control algorithmic structure

## Why not just PyTorch?
PyTorch ops are pre-fused only for common patterns. Custom ops (e.g. fused attention, custom activations) require either slow Python loops or writing CUDA. Triton gives you a third option.

## Installation & Setup
```bash
pip install torch
pip install triton

## Verify GPU and triton
python -c "import triton, import torch, print(torch.cuda.get_device_name(0))"
```

```python
import torch
import triton
import triton.language as tl

print(f"Triton version: {triton.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
```

## Core concepts
