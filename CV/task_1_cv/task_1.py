import cv2 as cv
import numpy as np

from Essentials import processing

img = cv.imread("ss.jpg", cv.IMREAD_GRAYSCALE)
img = img.astype(np.float64) / 255.0

k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")

proc = processing()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    -> the first test with k = 3 in part_1 !
A = np.random.rand(20, 20)
K = np.random.rand(k, k)
mine = proc.my_conv(A, K)
ref = cv.filter2D(A, -1, K, borderType=cv.BORDER_CONSTANT)
print(np.abs(mine - ref).max())  #  ->  8.881784197001252e-16
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    -> the first test with k = 5 in part_1 !
A = np.random.rand(20, 20)
K = np.random.rand(k, k)
mine = proc.my_conv(A, K)
ref = cv.filter2D(A, -1, K, borderType=cv.BORDER_CONSTANT)
print(np.abs(mine - ref).max())  #  ->  2.6645352591003757e-15
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    -> the first test with k = 7 in part_1 !
A = np.random.rand(20, 20)
K = np.random.rand(k, k)
mine = proc.my_conv(A, K)
ref = cv.filter2D(A, -1, K, borderType=cv.BORDER_CONSTANT)
print(np.abs(mine - ref).max())  #  ->  7.105427357601002e-15
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

cropped_box = proc.my_conv(img, proc.box_kernel(9))
cv.imshow("cropped_box", (cropped_box * 255.0).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/cropped_box.jpg",
    (cropped_box * 255.0).astype(np.uint8),
)


k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")

# ???????????????????????????????  -> part_2 gaussian filter for (σ = 1,k = 5)
mine = proc.gaussian_kernel(1, k)
g1d = cv.getGaussianKernel(k, 1)
ref = g1d @ g1d.T
print(np.abs(mine - ref).max())  # -> 1.3877787807814457e-17
# ???????????????????????????????

k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")


# ???????????????????????????????  -> part_2 gaussian filter for (σ = 2,k = 13)
mine = proc.gaussian_kernel(2, k)
g1d = cv.getGaussianKernel(k, 2)
ref = g1d @ g1d.T
print(np.abs(mine - ref).max())  # -> 2.0816681711721685e-17
# ???????????????????????????????


# EXPIREMENT #               -> part_2 gaussian filter 0,5 sigma !

print("k value for sigma = 0.5 !")
k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")


mine = proc.my_conv(img, proc.gaussian_kernel(0.5, k))
cv.imshow(f"sigma= 0.5  and k= {k}  !", (mine * 255.0).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/sigma=0.5,k={k}.jpg",
    (mine * 255.0).astype(np.uint8),
)
##########################    -> part_2 gaussian filter 1 sigma !
print("k value for sigma = 1 !")
k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")


mine = proc.my_conv(img, proc.gaussian_kernel(1, k))
cv.imshow(f"sigma= 1  and k= {k} !", (mine * 255.0).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/sigma=1,k={k}.jpg",
    (mine * 255.0).astype(np.uint8),
)
##########################      -> part_2 gaussian filter 2 sigma !
print("k value for sigma = 2 !")
k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")


mine = proc.my_conv(img, proc.gaussian_kernel(2, k))
cv.imshow(f"sigma= 2  and k= {k}  !", (mine * 255.0).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/sigma=2,k={k}.jpg",
    (mine * 255.0).astype(np.uint8),
)
##########################      -> part_2 gaussian filter 4 sigma !
print("k value for sigma = 4 !")
k = int(input("Enter the size of the kernal : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")


mine = proc.my_conv(img, proc.gaussian_kernel(4, k))
cv.imshow(f"sigma= 4  and k= {k}   !", (mine * 255.0).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/sigma=4,k={k}.jpg",
    (mine * 255.0).astype(np.uint8),
)
##########################


##########################  -> part_3 sanity check for DOG kernels !
k = int(input("Enter the size of the kernal for dog filters : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")

print(
    f"the total sum for the x_DOG_kernel is : {np.sum(proc.dog_kernel_x(0.5, k))}"
)  # -> -2.7755575615628914e-17
print(f"the array of the x_DOG_kernel is : {proc.dog_kernel_x(0.5, k)}")
print(
    f"the total sum for the y_DOG_kernel is : {np.sum(proc.dog_kernel_y(0.5, k))}"
)  # -> -> -2.7755575615628914e-17
print(f"the array of the y_DOG_kernel is : {proc.dog_kernel_y(0.5, k)}")
##########################


########################## -> part_3.2
k = int(input("Enter the size of the kernal for gradient dog filters : "))
if k % 2 == 0:
    raise ValueError("Enter an odd numbered kernal size !!")


Gx = proc.my_conv(img, proc.dog_kernel_x(0.5, k))
Gy = proc.my_conv(img, proc.dog_kernel_y(0.5, k))
magnitude = np.sqrt(Gx**2 + Gy**2)
direction = np.arctan2(Gy, Gx)

cv.imshow("Gx", (255.0 * np.abs(Gx)).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/Gx.jpg",
    (255.0 * np.abs(Gx)).astype(np.uint8),
)

cv.imshow("Gy", (255.0 * np.abs(Gy)).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/Gy.jpg",
    (255.0 * np.abs(Gy)).astype(np.uint8),
)

cv.imshow(
    "magnitude",
    (
        (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255.0
    ).astype(np.uint8),
)
cv.waitKey(1)
cv.imwrite(
    f"./task_images/magnitude.jpg",
    (
        (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255.0
    ).astype(np.uint8),
)
##########################


########################## -> part_3.3
noisy = img + np.random.normal(0, 0.05, img.shape)

sobel_x = proc.my_conv(noisy, proc.sobel_x())
cv.imshow("sobel_x_1", (255.0 * np.abs(sobel_x)).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/sobel_x_1.jpg",
    (255.0 * np.abs(sobel_x)).astype(np.uint8),
)

gaussian_blur = proc.my_conv(noisy, proc.gaussian_kernel(2, 13))
sobel_x = proc.my_conv(gaussian_blur, proc.sobel_x())
cv.imshow("sobel_x_2", (255.0 * np.abs(sobel_x)).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/sobel_x_2.jpg",
    (255.0 * np.abs(sobel_x)).astype(np.uint8),
)


dog_kernel = proc.my_conv(noisy, proc.dog_kernel_x(2, 13))
dog_abs = np.abs(dog_kernel)
dog_kernel = (dog_abs - dog_abs.min()) / (dog_abs.max() - dog_abs.min())
cv.imshow("dog_kernel", (255.0 * dog_kernel).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/dog_kernel.jpg",
    (255.0 * dog_kernel).astype(np.uint8),
)
##########################


########################## -> part_4.0

# part_1 :

# BOX :
mine = proc.my_conv(img, proc.box_kernel(5))
opencv = cv.blur(img, (5, 5))
print(
    f"box difference: {np.abs(mine - opencv).max()}"
)  # -> box difference: 0.552156862745098


# GAUSSIAN :
mine = proc.my_conv(img, proc.gaussian_kernel(2, 13))
opencv = cv.GaussianBlur(img, (13, 13), 2)
print(
    f"gaussian difference: {np.abs(mine - opencv).max()}"
)  # -> gaussian difference: 0.5523247505227877


# SOBEL_X :
mine = proc.my_conv(img, proc.sobel_x())
# 2. Apply Sobel X (dx=1, dy=0)
opencv = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=3)
print(
    f"sobel_x difference: {np.abs(mine - opencv).max()}"
)  # -> sobel_x difference: 3.7137254901960786


# SOBEL_Y :
mine = proc.my_conv(img, proc.sobel_y())
# 2. Apply Sobel Y (dx=0, dy=1)
opencv = cv.Sobel(img, cv.CV_8U, 0, 1, ksize=3)
print(
    f"sobel_y difference: {np.abs(mine - opencv).max()}"
)  # -> sobel_y difference: 3.537254901960784


#! Hint: check the borderType argument !!!
inner_diff = np.abs(mine[5:-5, 5:-5] - opencv[5:-5, 5:-5]).max()
print(
    f"Inner difference: {inner_diff}"
)  # -> Inner difference: 3.537254901960784 . to get values lower than 10e-6 , i should use the same border type of open_Cv and same displaying format !


# part_2 :

box = cv.blur(img, (5, 5))
cv.imshow("box", (box * 255.0).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/box.jpg",
    (box * 255.0).astype(np.uint8),
)

gauss = cv.GaussianBlur(img, (13, 13), 2)
cv.imshow("gauss", (gauss * 255.0).astype(np.uint8))
cv.waitKey(1)
cv.imwrite(
    f"./task_images/gauss.jpg",
    (gauss * 255.0).astype(np.uint8),
)

# part_3 :

# cv2.CV_64F captures negative gradients when moving from light to dark pixels
sobel_x_64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
# 3. Take absolute value and convert back to 8-bit (0-255)
sobel_x_64f = cv.convertScaleAbs(sobel_x_64f)
cv.imshow("sobel_x_64f", np.abs(sobel_x_64f) * 255.0)
cv.waitKey(1)
cv.imwrite(
    f"./task_images/sobel_x_64f.jpg",
    np.abs(sobel_x_64f) * 255.0,
)

sobel_x_8U = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=3)
# 3. Take absolute value and convert back to 8-bit (0-255)
sobel_x_8U = cv.convertScaleAbs(sobel_x_8U)
cv.imshow("sobel_x_8U", np.abs(sobel_x_8U) * 255.0)
cv.waitKey(1)
cv.imwrite(
    f"./task_images/sobel_x_8U.jpg",
    np.abs(sobel_x_8U) * 255.0,
)


cv.waitKey(0)
cv.destroyAllWindows()
