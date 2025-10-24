import math
import numpy as np
from MyConvolution import convolve
def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma:
float) -> np.ndarray:
"""
Create hybrid images by combining a low-pass and high-pass filtered pair.
:param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or
colour shape=(rows,cols,channels))
:type numpy.ndarray
:param lowSigma: the standard deviation of the Gaussian used for low-pass filtering
lowImage
:type float
:param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or
colour shape=(rows,cols,channels))
:type numpy.ndarray
:param highSigma: the standard deviation of the Gaussian used for low-pass filtering
highImage before subtraction to create the high-pass filtered image
:type float
:returns returns the hybrid image created
by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it
with
a high-pass image created by subtracting highImage from highImage convolved with
a Gaussian of s.d. highSigma. The resultant image has the same size as the input
images.
:rtype numpy.ndarray
"""
# Your code here.
def makeGaussianKernel(sigma: float) -> np.ndarray:
"""
Use this function to create a 2D gaussian kernel with standard deviation sigma.
The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
"""
# Your code here.
n = int(math.floor(8 * sigma + 1))
if n % 2 == 0:
    n += 1

radius = (n - 1) / 2
kernel = np.zeros((n, n), dtype=np.float32)

gaussian_const = 1 / (2 * math.pi * sigma * sigma)
gaussian_function = lambda x, y: gaussian_const * math.exp(-(x * x + y * y) / (2 * sigma * sigma))

total = 0.0
for i in range(n):
    for j in range(n):
        x = i - radius
        y = j - radius
        kernel[i, j] = gaussian_function(x, y)
        total += kernel[i, j]

for i in range(n):
    for j in range(n):
        kernel[i, j] /= total
return kernel
