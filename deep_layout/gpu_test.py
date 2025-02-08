import timeit

# CPU Function
def cpu_func():
    return sum(i**2 for i in range(10**7))

# GPU Function (Using Numba)
from numba import cuda
import numpy as np
@cuda.jit
def gpu_func(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = arr[idx] ** 2

arr = cuda.to_device(np.arange(10**7))
gpu_time = timeit.timeit(lambda: gpu_func[10**4, 1](arr), number=10)

cpu_time = timeit.timeit(cpu_func, number=10)

print(f"CPU Time: {cpu_time:.5f} sec")
print(f"GPU Time: {gpu_time:.5f} sec")
