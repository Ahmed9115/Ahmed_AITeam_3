import numpy as np


class processing:
    def my_conv(self, image, kernel, pad_mode="zero"):
        """
        image : 2D float array, shape (H, W)
        kernel : 2D float array, shape (k, k), k odd
        pad_mode : "zero"  | "edge" -> replicate | "reflect"
        returns : 2D float array, shape (H, W) <- same size as input
        """

        output = np.zeros_like(image)
        k = kernel.shape[1]
        pad_size = k // 2

        if pad_mode == "zero":
            pad_mode = "constant"
        elif pad_mode in ["edge", "replicate"]:
            pad_mode = "edge"
        elif pad_mode == "reflect":
            pad_mode = "reflect"
        else:
            print("Not available pad_mode , running the default (zero) !!")
            pad_mode = "constant"

        padded_image = np.pad(
            image, pad_width=pad_size, mode=pad_mode, constant_values=0
        )

        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                output[h, w] = np.sum(padded_image[h : h + k, w : w + k] * kernel)

        return output

    def box_kernel(self, k):
        kernel = np.ones((k, k)) / k**2
        return kernel

    def gaussian_kernel(self, sigma, k):
        if k % 2 == 0:
            raise ValueError("Enter an odd numbered kernal size !!")

        kernel = np.ones((k, k))
        for i in range(k):
            for j in range(k):
                kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(
                    -((j - k // 2) ** 2 + (i - k // 2) ** 2) / (2 * sigma**2)
                )
        kernel /= np.sum(kernel)
        return kernel

    def sobel_x(self):
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        return Gx

    def sobel_y(self):
        Gy = self.sobel_x().T
        return Gy

    def laplace_kernel(self):
        lp = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        return lp

    def dog_kernel_x(self, sigma, k):
        if k % 2 == 0:
            raise ValueError("Enter an odd numbered kernal size !!")

        kernel = np.ones((k, k))
        for i in range(k):
            for j in range(k):
                kernel[i, j] = (
                    (-(j - k // 2) / sigma**2)
                    * (1 / (2 * np.pi * sigma**2))
                    * np.exp(-((j - k // 2) ** 2 + (i - k // 2) ** 2) / (2 * sigma**2))
                )
        return kernel

    def dog_kernel_y(self, sigma, k):
        if k % 2 == 0:
            raise ValueError("Enter an odd numbered kernal size !!")

        kernel = np.ones((k, k))
        for i in range(k):
            for j in range(k):
                kernel[i, j] = (
                    (-(i - k // 2) / sigma**2)
                    * (1 / (2 * np.pi * sigma**2))
                    * np.exp(-((j - k // 2) ** 2 + (i - k // 2) ** 2) / (2 * sigma**2))
                )
        return kernel
