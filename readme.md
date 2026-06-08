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

## Core Concepts

## What is tile?
A small chunk/sub-part of data processed together by one GPU program.
* it can be:

1. a slice of an array
2. a sub-part of a matrix
3. even a small cube of tensor data

## 1D tile(Arrays):
```python
Large array: [1,2,3,4,5,6,7,8]
Tile 1:- [1,2,3,4]
Tile 2:- [5,6,7,8]
```

* Each Triton program processes one tile.

## 2D Tile(Matrix Block)
```python
[
 [1,2,3,4],
 [5,6,7,8],
 [9,10,11,12],
 [13,14,15,16]
]

Split into 2×2 tiles:

Tile A:
[1 2]
[5 6]

Tile B:
[3 4]
[7 8]
Tile C:
[ 9 10]
[13 14]

Tile D:
[11 12]
[15 16]
```

* Operations happen tile-by-tile.

## 3D Tile (Tensor Cube)

For deep learning tensors:--[BATCH, HEIGHT, WIDTH]
* a tile can be small cube/chunk of tensor data
* Example:-- 4x4x4 cube

## Why Tiles Exist

1. GPUs are optimized for many small parallel operations Instead of processing one element at a time
2. GPUs process entire tiles at once which is much faster.

## We will start with examples
```python
[
 [ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12],
 [13, 14, 15, 16]
]

BLOCK_M = 2
BLOCK_N = 2

Program 0:
[1 2]
[5 6]

Program 1:
[3 4]
[7 8]

Program 2:
[ 9 10]
[13 14]

Program 3:
[11 12]
[15 16]
```

## Program ID
worker handling one tile

```python
## each kernel launch spawns many programs. this is your block's unique index, similar to pythonblockIdx in CUDA
tl.program_id(axis)
```

## Block Pointer
How much data do I process?
```python
## creates a range of offsets within your block. you add this to a base pointer to address a tile of memory
tl.arange(0, BLOCK)
```

```python
pid = tl.program_id(0)

rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
cols = tl.arange(0, BLOCK_N)
```

## Load/Store
```python
## Go to this memory location and read values
eg. data = [10,20,30,40]
x = tl.load(ptr)
it reads 10
## Loading multiple values
offsets = [0,1,2,3]
x = tl.load(ptr+offsetc)
it reads [10,20,30,40]

tl.load(ptr, mask)
# why mask
offsets = [0,1,2,3]
data = [10,20,30]
without mask it try to read ptr+3 which is outside valid memory
So triton used:
n_elements = len(data)
mask = offsets < n_elements
mask = [True, True, True, False]
x = tl.load(ptr + offsets, mask=mask)

```
```python

## Write values back into memory
val = [2,4,6,8]
tl.store(pts, val, mask)
so it writes [2,4,6,8] into GPU memory


tl.store(ptr + offsets, x, mask=mask)
```

## Constexpr
```python
## block sizes must be compile-time constants (powers of 2). This allows the compiler to fully unroll loops and optimize memory access.
tl.constexpr
```

```python
@triton.jit
def my_kernel(in_ptr, out_ptr, n, BLOCK:tl.constexpr):
    pid = tl.program_id(0)            # which block am I?
    offsets = pid * BLOCK + tl.arange(0, BLOCK)     # block's indices
    mask  = offsets < n            # guard out-of-bounds
    x = tl.load(in_ptr + offsets, mask)      # read global memory
    tl.store(out_ptr + offsets, x*2.0, mask)    # write global memory
```

## Lets write out first kernel : Vector Addition
```python
import torch
import triton
import triton.language as tl

@trition.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offests, mask=mask)
    y = tl.load(y_ptr + offests, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask)

def add(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda
    n_elements = output.numel()

    # grid = number of programs to launch
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid][x,y,output, n_elements, BLOCk_SIZE=1024]
    return output
```

```python
# Testing
torch.manual_seed(0)
a = torch.rand(98432, device='cuda')
b = torch.rand(98432, device='cuda')
triton.output = add(a,b)
torch.output = a + b

print(f"Max diff: {(triton_output - torch_output).abs().max():.6f}")
```

## Memory Layout & Pointer arithmatic
Efficient memory access is what separates fast kernels from slow ones. Triton gives you tools to express 2D tile addressing, strided access, and coalesced reads.

```python
[
    [1,2,3],
    [4,5,6]
]

Stored in memory as:
[1,2,3,4,5,6]
This is called row-major layout

Example:

ptr + 0
ptr + 1
ptr + 2

Used to access tensor elements.
```

## Pointer arithmatic
```python
ptr + row * stride + col
where:
row = row_index
col = col_index
stride = number of elements per row
```
```python
[
    [1,2,3],
    [4,5,6]
]
to access matrix[1][2] we use this formula (ptr + row * stride + col) = ptr + 1 * 3 + 2
which is points to 6
```

```python
## python - 2D pointer blocks
@triton.jit
def matrix_kernel(A_ptr, B_ptr, stride_am, stride_an, M, N, BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr):
       pid_m = tl.program_id(0)
       pid_n = tl.program_id(1)

       row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
       col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

       ptrs = (A_ptr + row_offset[:, None] * stride_am + col_offset[None, :] * stride_an)

       mask = (row_offset[:, None] < M) & (col_offset[None, :] < N)
       tile = tl.load(ptrs, mask=mask, other=0.0)
       tl.store(B_ptr + row_offset[:, None] * stride_am + col_offset[None, :] *  stride_an , tile * 2.0, mask=mask)

```

```python
#using tl.make_block_ptr()

a_block_ptr = tl.make_block_ptr(
    base=A_ptr,
    shape=(M, N),
    strides=(stride_am, stride_an),
    offsets=(pid_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1,0)
)

a = tl.load(a_block_ptr)              # no mask needed!
a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))  # move right
```

## Reductions
Triton provides built-in reduction operations that map to efficient warp-level and block-level reduce instructions. Use these instead of manual loops.

```python
@triton.jit

def reduction_demo(x_ptr, out_ptr, M, N, Block_N:tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=None)

    row_max = tl.max(x, axis=0) # max across axis 0
    row_sum = tl.sum(x, axis=0) # sum across axis 0
    row_min = tl.min(x, axis=0) # min across axis 0

    x_shifted = x - row_max
    log_sum_exp = tl.log(tl.sum(tl.exp(x_shifted), axis=0)) + row_max

    tl.store(out_ptr + row, log_sum_exp)
```

```python
@triton.jit
def row_col_reduce(X_ptr, row_out, col_out, M, N,
                   BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    x = tl.load(X_ptr + rows[:, None] * N + cols[None, :])

    # axis=1: reduce across columns → shape [BM]
    row_sums = tl.sum(x, axis=1)
    # axis=0: reduce across rows → shape [BN]
    col_sums = tl.sum(x, axis=0)
    tl.store(row_out + rows, row_sums)
    tl.store(col_out + cols, col_sums)
```
## Matrix multiplication(GEMM)
Matrix multiply is the most important kernel in deep learning. The Triton approach tiles both A and B, accumulating partial products in SRAM. This achieves near-cuBLAS performance with far less code

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    Block_M:tl.constexpr, Block_N:tl.constexpr, Block_K:tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offset = pid * Block_M + tl.arange(0, Block_M)
    n_offset = pid * Block_N + tl.arange(0, Block_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, Block_K)):
        k_offsets = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptr + m_offset[:, None] * strided_am + k_offsets[None, :] * strided_ak, mask=a_mask, other=0.0)

        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn, mask=b_mask, other=0.0)

        # Tensor-core matrix multiply-accumulate
        acc += tl.dot(a, b)

    # output tile
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptr + m_offsets[:, None] * stride_cm
                    + n_offsets[None, :] * stride_cn,
             acc, mask=c_mask)


def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )

    return c

```