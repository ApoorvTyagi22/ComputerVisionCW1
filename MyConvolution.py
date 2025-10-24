import numpy as np
def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
"""
Convolve an image with a kernel assuming zero-padding of the image to handle the borders
:param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
:type numpy.ndarray
:param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
:type numpy.ndarray
:returns the convolved image (of the same shape as the input image)
:rtype numpy.ndarray
"""
# Your code here. You'll need to vectorise your implementation to ensure it runs
# at a reasonable speed.

if len(image.shape) == 3:
    # Colour image
    image_height, image_width, num_channels = image.shape
    output = np.zeros((image_height, image_width, num_channels), dtype=image.dtype)
else: 
    # Greyscale image
    image_height, image_width = image.shape[:2]
    output = np.zeros((image_height, image_width), dtype=image.dtype)


kernel_height, kernel_width = kernel.shape

pad_height = (kernel_height - 1) // 2
pad_width = (kernel_width - 1) // 2

if num_channels is None:
    num_channels = 1
    
# Channel loop
for c in range(num_channels):
    for i in range(image_height):
        for j in range(image_width):
            sum = 0.0

            # Kernel loop
            for k in range(-pad_height, pad_height + 1):
                for l in range(-pad_width, pad_width + 1):
                    i_dash = i - k # convolution operation and not correlation
                    j_dash = j - l

                    if 0 <= i_dash < image_height and 0 <= j_dash < image_width:
                        if num_channels == 1:
                            sum += image[i_dash, j_dash] * kernel[k + pad_height, l + pad_width]
                        else:
                            sum += image[i_dash, j_dash, c] * kernel[k + pad_height, l + pad_width]

            if num_channels == 1:
                output[i, j] = sum
            else:
                output[i, j, c] = sum

return output     
                    