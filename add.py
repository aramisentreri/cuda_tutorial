import numpy as np
from numba import vectorize
import timeit


@vectorize(['float32(float32, float32)'], target='cpu') # replace target with "cuda" in a GPU compatible machine
def Add(a, b):
  return a + b

# Initialize arrays
N = 100000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)

# Add arrays on GPU
start_time = timeit.default_timer()
C = Add(A, B)
elapsed = timeit.default_timer() - start_time
print(f"Using vectorize, execution time is {elapsed}")


start_time = timeit.default_timer()
D = A + B
elapsed = timeit.default_timer() - start_time
print(f"Using vectorize, execution time is {elapsed}")

