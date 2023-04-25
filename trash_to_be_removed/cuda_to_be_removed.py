from numba import cuda, float32
import torch
import time
import cupy as cp
import numpy as np

'''
GPU: NVIDIA GeForce GTX 1080 Ti'
According to specifications on this device 
(https://www.nvidia.com/en-gb/geforce/graphics-cards/geforce-gtx-1080-ti/specifications/),
we have:
CUDA Cores: 3584

'''

# This make us run it on the GPU (quite fast so to say)


# @cuda.jit
# def matmul(A, B, C):
#     """Perform square matrix multiplication of C = A * B
#     """
#     i, j = cuda.grid(2)
#     if i < C.shape[0] and j < C.shape[1]:
#         tmp = 0.
#         for k in range(A.shape[1]):
#             tmp += A[i, k] * B[k, j]
#         C[i, j] = tmp

def test_cuda(pc0, pc1):
    # Code to remove

    # A = torch.swapaxes(pc0, 0, 1).numpy()
    # B = pc1.numpy()
    N = 2000
    A = torch.rand(N, N, dtype=torch.float64).numpy()
    B = torch.rand(N, N, dtype=torch.float64).numpy()
    A_cp = cp.asarray(A)
    B_cp = cp.asarray(B)
    A_dev = cuda.to_device(A)
    B_dev = cuda.to_device(B)

    t1_ord = time.time()
    F = A@B
    print(A.shape, B.shape, F.shape)
    print(f"numpy matrix multiplication {round(time.time()-t1_ord, 5)}")

    TPB = 16
    threadsperblock = (TPB, TPB)  # each block will contain 16x16 threads
    C = cp.zeros((N, N), dtype=np.float64)
    D = cp.zeros((N, N), dtype=np.float64)
    blockpergrid_x = int(np.ceil(C.shape[0] / threadsperblock[0]))
    blockpergrid_y = int(np.ceil(C.shape[1] / threadsperblock[1]))
    blockspergrid = (blockpergrid_x, blockpergrid_y)
    print(blockspergrid)
    print(f"The kernel will be executed up to element {threadsperblock[0]*blockpergrid_x}")
    t1_ord = time.time()
    matmul[blockspergrid, threadsperblock](A_cp, B_cp, C)
    print("matmul + compilation: ", round(time.time()-t1_ord, 5))
    t1_ord = time.time()
    matmul[blockspergrid, threadsperblock](A_cp, B_cp, D)
    print("matmul:", round(time.time()-t1_ord, 5))
    assert np.allclose(F, C.get()), "Cuda.jit matmul not working"

    threadsperblock = (TPB, TPB)  # each block will contain 16x16 threads
    C = cp.zeros((N, N), dtype=np.float64)
    D = cp.zeros((N, N), dtype=np.float64)
    blockpergrid_x = int(np.ceil(C.shape[0] / threadsperblock[0]))
    blockpergrid_y = int(np.ceil(C.shape[1] / threadsperblock[1]))
    blockspergrid = (blockpergrid_x, blockpergrid_y)
    print(blockspergrid)
    print(f"The kernel will be executed up to element {threadsperblock[0]*blockpergrid_x}")
    t1_ord = time.time()
    fast_matmul[blockspergrid, threadsperblock](A_dev, B_dev, C)
    print("fast_matmul + compilation", round(time.time()-t1_ord, 5))
    time.sleep(3)
    t1_ord = time.time()
    fast_matmul[blockspergrid, threadsperblock](A_cp, B_cp, D)
    print("fast_matmul", round(time.time()-t1_ord, 5))
    time.sleep(3)
    t1_ord = time.time()
    fast_matmul[blockspergrid, threadsperblock](A_dev, B_dev, D)
    print("fast_matmul 2nd", round(time.time()-t1_ord, 5))
    assert np.allclose(F, C.get()), "Cuda.jit fast_matmul not working"
    assert np.allclose(F, D.get()), "Cuda.jit fast_matmul not working"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_torch = torch.tensor(A, dtype=torch.float64).to(device)
    B_torch = torch.tensor(B, dtype=torch.float64).to(device)

    t1_ord = time.time()
    C = A_torch@B_torch
    print("torch @", round(time.time()-t1_ord, 5))
    assert np.allclose(F, C.cpu().numpy()), "torch @ not working"

    t1_ord = time.time()

    D = torch.matmul(A_torch, B_torch)
    print("torch matmul", round(time.time()-t1_ord, 5))
    t1_ord = time.time()
    assert np.allclose(F, D.cpu().numpy()), "torch @ not working"

    t1_ord = time.time()
    D = cp.dot(A, B)
    print("cupy dot", round(time.time()-t1_ord, 5))
    assert np.allclose(F, D), "cupy dot not working"
    t1_ord = time.time()
    D = cp.dot(A_cp, B_cp)
    print("cupy dot cp", round(time.time()-t1_ord, 5))
    assert np.allclose(F, D.get()), "cupy dot not working"
    exit()
    # Test increment by 1
    # threadsperblock = 32
    # blockspergrid = (torch.numel(A) + (threadsperblock - 1)) // threadsperblock
    # threadsperblock = 1024
    # # J = F.flatten()
    # blockspergrid = (A.size + (threadsperblock - 1)) // threadsperblock
    # A2 = A.copy()

    # t1_ord = time.time()
    # U = A + 1
    # print("Numpy", round(time.time()-t1_ord, 5))

    # # t1_ord = time.time()
    # # increment_by_one[blockspergrid, threadsperblock](J)
    # # print("Numba1", round(time.time()-t1_ord, 5))

    # t1_ord = time.time()
    # increment_by_one_alt2[blockspergrid, threadsperblock](A)
    # print("Numba2", round(time.time()-t1_ord, 5))

    # t1_ord = time.time()
    # increment_by_one_alt2[blockspergrid, threadsperblock](A2)
    # print("Numba2", round(time.time()-t1_ord, 5))

    # # assert np.allclose(J, U), "Errror"
    # assert np.allclose(A2, U), "Errror"

    # Test matrix multiplication
    # C = torch.empty_like(torch.tensor(F)).numpy()


@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1


@cuda.jit
def increment_by_one_alt2(A):
    row, column = cuda.grid(2)
    print(row, column)
    if row < A.shape[0] and column < A.shape[1]:
        A[row, column] += 1

# @cuda.jit
# def add(a, b, c):
#     """Perform vector addition c = a + b
#     """
#     i, j = cuda.grid(2)
#     c = a + b

# This is not working, but gives us machine code
# @jit
# def matmul(A, B, C):
#     """Perform square matrix multiplication of C = A * B
#     """
#     for i in range(len(A)):
#         # iterate through columns of Y
#         for j in range(len(B[0])):
#             # iterate through rows of Y
#             for k in range(len(B)):
#                 C[i][j] += A[i][k] * B[k][j]
#     return C


@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


TPB = 16  # threads per block


@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp
