# Triton:GPU Programming A to Z

## What is Triton
Triton is open-source programming language and compiler developed by OPENAI. it lets you write high-performance CUDA kernels in python. triton sits betweeen CUDA(too low-level) and pytorch(too high-level)

## Why not raw CUDA
CUDA requires managing thread blocks, warps shared memory layouts, bank conflicts, and register pressure manually. Triton automates these while letting you control algorithmic structure

## Why not just PyTorch
PyTorch ops are pre-fused only for common patterns. Custom ops (e.g. fused attention, custom activations) require either slow Python loops or writing CUDA. Triton gives you a third option.