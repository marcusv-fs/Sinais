import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('one_piece_1.jpg', cv.IMREAD_GRAYSCALE)
dft = np.fft.fft2(img)
dftShift = np.fft.fftshift(dft)

inv_dftShift = np.fft.ifftshift(dftShift)
inv_dft = np.fft.ifft2(inv_dftShift)


# plt.imshow(np.float32(inv_dft), cmap="gray")
# plt.show()

plt.subplot(131),plt.imshow(np.float32(img), cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(np.float32(dftShift), cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(np.float32(inv_dft), cmap = 'gray')
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()
