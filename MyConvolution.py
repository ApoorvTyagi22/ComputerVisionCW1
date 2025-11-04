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
        num_channels = 1
        output = np.zeros((image_height, image_width), dtype=image.dtype)


    kernel_height, kernel_width = kernel.shape

    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2

    kernel = kernel[::-1, ::-1]

    # Convert to float for math
    img = image.astype(np.float64, copy=False)

    # zero-padding
    if num_channels == 1:
        img_padded = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width), dtype=np.float64)
        img_padded[pad_height:pad_height + image_height, pad_width:pad_width + image_width] = img
    else:
        img_padded = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width, num_channels), dtype=np.float64)
        img_padded[pad_height:pad_height + image_height, pad_width:pad_width + image_width, :] = img



    # Channel loop
    for c in range(num_channels):
        for i in range(image_height):
            for j in range(image_width):
                if num_channels == 1:
                    region = img_padded[i:i + kernel_height, j:j + kernel_width]
                else:
                    region = img_padded[i:i + kernel_height, j:j + kernel_width, c]
                val = np.sum(region * kernel)

                if num_channels == 1:
                    output[i, j] = val
                else:
                    output[i, j, c] = val
        

    return output.astype(image.dtype)