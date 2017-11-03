# Description: Gibbs sampling for image denoising
# Author: Zhu Lei
# Email: zlheui2@gmail.com
# Date: 3/11/2017

import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python
import math
import sys

def read_data(filename, is_RGB, visualize=False, save=False, save_name=None):
# read the text data file
#   data, image = read_data(filename, is_RGB) read the data file named 
#   filename. Return the data matrix with same shape as data in the file. 
#   If is_RGB is False, the data will be regarded as Lab and convert to  
#   RGB format to visualise and save.
#
#   data, image = read_data(filename, is_RGB, visualize)  
#   If visualize is True, the data will be shown. Default value is False.
#
#   data, image = read_data(filename, is_RGB, visualize, save)  
#   If save is True, the image will be saved in an jpg image with same name
#   as the text filename. Default value is False.
#
#   data, image = read_data(filename, is_RGB, visualize, save, save_name)  
#   The image filename.
#
#   Example: data, image = read_data("1_noise.txt", True)
#   Example: data, image = read_data("cow.txt", False, True, True, "segmented_cow.jpg")

	with open(filename, "r") as f:
		lines = f.readlines()

	data = []

	for line in lines:
		data.append(list(map(float, line.split(" "))))

	data = np.asarray(data).astype(np.float32)

	N, D = data.shape

	cols = int(data[-1, 0] + 1)
	rows = int(data[-1, 1] + 1)
	channels = D - 2
	img_data = data[:, 2:]

	# In numpy, transforming 1d array to 2d is in row-major order, which is different from the way image data is organized.
	image = np.reshape(img_data, [cols, rows, channels]).transpose((1, 0, 2))

	if visualize:
		if channels == 1:
			# for visualizing grayscale image
			cv2.imshow("", image)
		else:
			# for visualizing RGB image
			cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_Lab2BGR))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if save:
		if save_name is None:
			save_name = filename[:-4] + ".jpg"
		assert save_name.endswith(".jpg") or save_name.endswith(".png"), "Please specify the file type in suffix in 'save_name'!"

		if channels == 1:
			# for saving grayscale image
			cv2.imwrite(save_name, image)
		else:
			# for saving RGB image
			cv2.imwrite(save_name, (cv2.cvtColor(image, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))

	if not is_RGB:
		image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

	return data, image

def write_data(data, filename):
# write the matrix into a text file
#   write_data(data, filename) write 2d matrix data into a text file named
#   filename.
#
#   Example: write_data(data, "cow.txt")

	lines = []
	for i in range(len(data)):
		lines.append(" ".join([str(int(data[i, 0])), str(int(data[i, 1]))] + ["%.6f" % v for v in data[i, 2:]]) + "\n")

	with open(filename, "w") as f:
		f.writelines(lines)



filename = sys.argv[1]
data, image = read_data(filename, True)

J = 3
burnin = 100
loops = 1001

h, w, _ = image.shape

current_state = np.empty(image.shape)

# initialize the state
for i in range(0, h):
	for j in range(0, w):
		if image[i, j, 0] == 0:
			current_state[i, j, 0] = -1
		else:
			current_state[i, j, 0] = 1


# avg_state record the average of all sampled states
avg_state = np.zeros(image.shape)

# Gibbs sampling, sampling burnin+loops iterations, and we only use samples after burnin iterations
for k in range(0, burnin+loops):
	for i in range(0, h):
		for j in range(0, w):

			# calculate the exponent of pairwise potential
			pairwise_one_exponent = 0
			pairwise_neg_one_exponent = 0
			if j - 1 >= 0:
				pairwise_one_exponent += current_state[i, j-1, 0]
				pairwise_neg_one_exponent -= current_state[i, j-1, 0]
			if j + 1 < w:
				pairwise_one_exponent += current_state[i, j+1, 0]
				pairwise_neg_one_exponent -= current_state[i, j+1, 0]
			if i - 1 >= 0:
				pairwise_one_exponent += current_state[i-1, j, 0]
				pairwise_neg_one_exponent -= current_state[i-1, j, 0]
			if i + 1 < h:
				pairwise_one_exponent += current_state[i+1, j, 0]
				pairwise_neg_one_exponent -= current_state[i+1, j, 0]

			prob_pixel_one = 0
			prob_pixel_neg_one = 0

			# combine the local evidence term into the posterior distribution
			if image[i, j, 0] == 0:
				prob_pixel_one = math.exp(J*pairwise_one_exponent - pow(1-(-1), 2))
				prob_pixel_neg_one = math.exp(J*pairwise_neg_one_exponent - pow(-1-(-1), 2))
			else:
				prob_pixel_one = math.exp(J*pairwise_one_exponent - pow(1-1, 2))
				prob_pixel_neg_one = math.exp(J*pairwise_neg_one_exponent - pow(-1-1, 2))

			# calculate the probability of the state being +1
			prob_pixel_one = prob_pixel_one / (prob_pixel_one + prob_pixel_neg_one)

			if np.random.uniform() > prob_pixel_one:
				current_state[i, j, 0] = -1
			else:
				current_state[i, j, 0] = 1


	print("loops: " + str(k+1))
	# only start recording the sampling information after the burnin period
	if k >= burnin:
		
		avg_state += current_state


# denoising the image using Gibbs sampling information
denoised_image = np.empty(image.shape)
for i in range(0, h):
	for j in range(0, w):
		if avg_state[i, j, 0] > 0:
			denoised_image[i, j, 0] = 255
		else:
			denoised_image[i, j, 0] = 0


# display image
# cv2.imshow("", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow("", denoised_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# output image
cv2.imwrite(filename.split("_")[0]+"_denoise.png", denoised_image)
