# Description: EM algorithm for Gaussian Mixture model to do image segmentation
# Author: Zhu Lei
# Email: zlheui2@gmail.com
# Date: 27/10/2017

import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python
import sys
import math


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
		image = cv2.cvtColor(image, cv2.COLOR_Lab2BGR)

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


def pdf_multivariate_gauss(x, mu, cov):
# Caculate the multivariate normal density (pdf)
# Keyword arguments:
# 	x = numpy array of a "d x 1" sample vector
# 	mu = numpy array of a "d x 1" mean vector
# 	cov = "numpy array of a d x d" covariance matrix
   
    # evaluate the pdf of multi-variate gaussian distribution using its formula
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    
    return float(part1 * np.exp(part2))


k = 2
def initialize(data):
# initialize parameters miu, sigma, pi for Gaussian Mixture model
# each component of miu[i] (i = 0,...,k-1) is randomly initialized within 40% to 60% of range of the whole data
# sigma[i] (i = 0,...,k-1) is initialized as the covariance matrix of the whole data
	
	_, w = data.shape

	num_feature = w - 2

	# features_max stores the maximum value for each feature
	features_max = np.empty([num_feature])
	# features_min stores the minimum value for each feature
	features_min = np.empty([num_feature])
	for i in range(2, w):
		features_max[i-2] = max(data[:, i])
		features_min[i-2] = min(data[:, i])

	# features_range stores the range between the maximum value and minimum value for each feature
	features_range = features_max - features_min

	# calculate the covariance of the features for the whole data, and use it to initialize sigma
	features_cov = np.cov(data[:, 2:w].transpose())

	# initialize miu and sigma for Gaussian distribution
	miu = np.empty([k, num_feature])
	sigma = np.empty([k, num_feature, num_feature])
	for i in range(0, k):
		miu[i] = features_min + np.random.uniform(0.4, 0.6) * features_range
		sigma[i] = features_cov
		# make sure the covariance matrix is positive-definite, otherwise we can not calculate its pdf
		sigma[i] += (sigma[i,0,0] * 0.1) * np.identity(num_feature)

	# initialize pi vector
	pi = np.random.rand(k)
	pi = pi / np.sum(pi)
	# make sure the sum of the pi vector is 1
	pi[-1] = 1- np.sum(pi[0:k-1])

	return miu, sigma, pi

def evaluate(data, miu, sigma, pi):
# evaluate the log likelihood

	n, w = data.shape

	likelihood = 0
	for i in range(0, n):
		accumulate = 0
		for j in range(0, k):
			accumulate += pi[j] * pdf_multivariate_gauss(data[i, 2:w], miu[j], sigma[j])
		likelihood += np.log(accumulate)

	return likelihood

def expectation(data, miu, sigma, pi):
# evaluate the responsibilities using the current parameter values

	n, w = data.shape
	gamma = np.empty([n, k])

	for i in range(0, n):
		for j in range(0, k):
			gamma[i, j] = pi[j] * pdf_multivariate_gauss(data[i, 2:w], miu[j], sigma[j])
		gamma[i, :] = gamma[i, :] / np.sum(gamma[i, :])
	return gamma


def maximization(data, gamma):
# re-estimate the parameters using the current responsibilities

	n, w = data.shape

	num_feature = w - 2

	N = np.empty([k])
	for i in range(0, k):
		N[i] = np.sum(gamma[:,i])

	miu_update = np.empty([k, num_feature])
	sigma_update = np.empty([k, num_feature, num_feature])

	for i in range(0, k):
		# update miu[i]
		miu_update[i] = np.sum(np.multiply(data[:, 2:w], gamma[:,i].reshape(n,1)), axis=0) / N[i]
		# update sigma[i]
		sigma_tmp = np.zeros([num_feature, num_feature])
		for j in range(0, n):
			sigma_tmp = sigma_tmp + gamma[j, i] * np.outer(data[j, 2:w] - miu_update[i], data[j, 2:w] - miu_update[i])
		sigma_update[i] = sigma_tmp / N[i]

		# make sure the covariance matrix is positive-definite, otherwise we can not calculate its pdf
		sigma_update[i] += (sigma_update[i,0,0] * 0.1) * np.identity(num_feature)

	# update pi
	pi_update = N / n 

	return miu_update, sigma_update, pi_update


def cal_neighboring_diff(data, idx, h, w):
# calculate the sum of difference between a pixel and its neighboring pixels

	diff = np.zeros([h*w])
	for i in range(0, h):
		for j in range(0, w):
			# make sure the index for the left most neighbor is larger than 0
			neighbor_left = max(j-1, 0)
			# make sure the index for the right most neighbor is smaller than w
			neighbor_right = min(j+1, w-1)
			# make sure the index for the upper most neighbor is larger than 0
			neighbor_up = max(i-1, 0)
			# make sure the index for the lower most neighbor is smaller than h
			neighbor_down = min(i+1, h-1)

			# calculate the difference between pixel (i, j) with its neighbors
			for ii in range(neighbor_up, neighbor_down+1):
				for jj in range(neighbor_left, neighbor_right+1):
					diff[j*h+i] += abs(data[j*h+i, idx] - data[jj*h+ii, idx])

	return diff

def cal_neighboring_avg(data, idx, h, w, step):
# calculate the average of a pixel's neighboring pixels
# step: step controls the size of the window we will consider for the neighborhood, 
#       for example, step = 1, we will consider the 3*3 window centered at each pixel; 
#                    step = 2, we will consider the 5*5 window centered at each pixel;
#                    step = k, we will consider the (2k+1)*(2k+1) window centered at each pixel.

	avg = np.zeros([h*w])
	for i in range(0, h):
		for j in range(0, w):
			# make sure the index for the left most neighbor is larger than 0
			neighbor_left = max(j-step, 0)
			# make sure the index for the right most neighbor is smaller than w
			neighbor_right = min(j+step, w-1)
			# make sure the index for the upper most neighbor is larger than 0
			neighbor_up = max(i-step, 0)
			# make sure the index for the lower most neighbor is smaller than h
			neighbor_down = min(i+step, h-1)

			# calculate the difference between pixel (i, j) with its neighbors
			for ii in range(neighbor_up, neighbor_down+1):
				for jj in range(neighbor_left, neighbor_right+1):
					avg[j*h+i] += data[jj*h+ii, idx]

			avg[j*h+i] /= (neighbor_right - neighbor_left + 1 + neighbor_down - neighbor_up + 1)

	return avg


def add_extra_features(data, h, w):
# adding extra features to the data set
	
	# adding the sum of difference of L, a, b value between each pixel and its neighboring pixels as a new feature
	# Lab_diff = np.empty([h*w, 3])

	# adding the sum of difference of L
	# Lab_diff[:, 0] = cal_neighboring_diff(data, 2, h, w)
	# adding the sum of difference of a
	# Lab_diff[:, 1] = cal_neighboring_diff(data, 3, h, w)
	# adding the sum of difference of b
	# Lab_diff[:, 2] = cal_neighboring_diff(data, 4, h, w)


	# adding the average of L, a, b value as new feature
	Lab_avg = np.empty([h*w, 3])

	# adding the neighboring average of L
	Lab_avg[:, 0] = cal_neighboring_avg(data, 2, h, w, 1)
	# adding the neighboring average of a
	Lab_avg[:, 1] = cal_neighboring_avg(data, 3, h, w, 1)
	# adding the neighboring average of b
	Lab_avg[:, 2] = cal_neighboring_avg(data, 4, h, w, 1)


	return np.append(data, Lab_avg, axis=1)




filename = sys.argv[1]
data, image = read_data(filename, False)

h, w, _ = image.shape

# adding extra features to the data
# data = add_extra_features(data, h, w)

# initialize parameters
miu, sigma, pi = initialize(data)
miu_update = np.copy(miu)
sigma_update = np.copy(sigma)
pi_update = np.copy(pi)

count = 1
# EM iterations
while True:

	# evaluate responsibilities
	gamma = expectation(data, miu_update, sigma_update, pi_update)
	
	miu = np.copy(miu_update)
	sigma = np.copy(sigma_update)
	pi = np.copy(pi_update)
	# re-estimate parameters
	miu_update, sigma_update, pi_update = maximization(data, gamma)

	# diff_norm stores the difference between consecutive parameters, it uses 2-norm for miu, square of frobenius norm for sigma, 100 times 1-norm for pi
	diff_norm = 0
	for i in range(0, k):
		diff_norm += np.linalg.norm(miu_update[i] - miu[i])
		diff_norm += math.pow(np.linalg.norm(sigma_update[i] - sigma[i]), 2)
		diff_norm += 100*abs(pi_update[i] - pi[i])

	# output runtime information
	print("Iteration " + str(count) + ": ")
	print(str(diff_norm) + "\n")
	count += 1

	# if the change of parameters is small, then output the result and break EM iterations
	if diff_norm < 1e-08:

		# mask image
		h, w, _ = image.shape
		mask = np.ones(image.shape, np.float32)
		foreground = np.copy(image)
		background = np.copy(image)
		for i in range(0, w):
			for j in range(0, h):
				if gamma[i*h+j, 0] > gamma[i*h+j, 1]:
					mask[j,i] = np.array([0,0,0])
					background[j,i] = np.array([0,0,0])
				else:
					foreground[j,i] = np.array([0,0,0])

		# display the image
		# cv2.imshow("", mask)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# cv2.imshow("", foreground)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# cv2.imshow("", background)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# output
		cv2.imwrite(filename.split(".")[0]+"_mask.jpg", mask*255)
		cv2.imwrite(filename.split(".")[0]+"_seg1.jpg", foreground*255)
		cv2.imwrite(filename.split(".")[0]+"_seg2.jpg", background*255)

		break
		