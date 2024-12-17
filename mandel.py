import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import jit
from numba import cuda
from numba import *


@jit
def mandel(x, y, max_iters):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters

@jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]
  
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
      
    for x in range(width):
      real = min_x + x * pixel_size_x
      for y in range(height):
        imag = min_y + y * pixel_size_y
        color = mandel(real, imag, iters)
        image[y, x] = color


image = np.zeros((1024, 1536), dtype = np.uint8)
start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) 
dt = timer() - start

print(f"Mandelbrot created in {dt} s")
imshow(image)
show()


# Even bigger speedups with CUDA

mandel_gpu = cuda.jit(device=True)(mandel)
 

@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x)/width
    pixel_size_y = (max_y - min_y)/height

    # threads in blocks

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    print(f"Dimensions: startX = {startX}, startY = {startY}, gridX = {gridX}, gridY = {gridY}")
    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y,x] = mandel_gpu(real, imag, iters)


gimage = np.zeros((1024, 1536), dtype = np.uint8)
blockdim = (32, 8)
griddim = (32, 16)

start = timer()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20)
d_image.copy_to_host()
dt = timer() - start

print(f"Mandelbrot created on GPU in {dt} s")

imshow(gimage)
show()

