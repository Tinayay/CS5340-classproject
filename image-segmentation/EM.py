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
# each component of miu[i] (i = 0,...,k-1) is randomly initialized within the range of the whole data
# sigma[i] (i = 0,...,k-1) is randomly initialized as a fraction times the covariance matrix of the whole data, the fraction is randomly chosen from 1/(2k) to 1/k

	# find the maximum and minimum value for L, a, b component
	L_max = max(data[:, 2])
	L_min = min(data[:, 2])
	a_max = max(data[:, 3])
	a_min = min(data[:, 3])
	b_max = max(data[:, 4])
	b_min = min(data[:, 4])

	# vector for Lab max and min value
	Lab_max = np.array([L_max, a_max, b_max])
	Lab_min = np.array([L_min, a_min, b_min])
	Lab_range = Lab_max - Lab_min

	# calculate the covariance of L, a, b for the whole data, and use it to initialize sigma
	Lab_cov = np.cov(data[:, 2:5].transpose())

	# initialize miu and sigma for Gaussian distribution
	miu = np.empty([k, 3])
	sigma = np.empty([k, 3, 3])
	for i in range(0, k):
		miu[i, :] = Lab_min + np.random.uniform() * Lab_range
		sigma[i, :, :] = np.random.uniform(1.0/(2*k), 1.0/k) * Lab_cov


	# initialize pi vector
	pi = np.random.rand(k)
	pi = pi / np.sum(pi)
	# make sure the sum of the pi vector is 1
	pi[-1] = 1- np.sum(pi[0:k-1])

	return miu, sigma, pi

def evaluate(data, miu, sigma, pi):
# evaluate the log likelihood

	n, _ = data.shape

	likelihood = 0
	for i in range(0, n):
		accumulate = 0
		for j in range(0, k):
			accumulate += pi[j] * pdf_multivariate_gauss(data[i, 2:5], miu[j], sigma[j])
		likelihood += np.log(accumulate)

	return likelihood

def expectation(data, miu, sigma, pi):
# evaluate the responsibilities using the current parameter values

	n, _ = data.shape
	gamma = np.empty([n, k])

	for i in range(0, n):
		for j in range(0, k):
			gamma[i, j] = pi[j] * pdf_multivariate_gauss(data[i, 2:5], miu[j], sigma[j])
		gamma[i, :] = gamma[i, :] / np.sum(gamma[i, :])

	return gamma


def maximization(data, gamma):
# re-estimate the parameters using the current responsibilities

	n, _ = data.shape

	N = np.empty([k])
	for i in range(0, k):
		N[i] = np.sum(gamma[:,i])

	miu_update = np.empty([k, 3])
	sigma_upate = np.empty([k, 3, 3])

	for i in range(0, k):
		# update miu[i]
		miu_update[i] = np.sum(np.multiply(data[:, 2:5], gamma[:,i].reshape(n,1)), axis=0) / N[i]
		# update sigma[i]
		sigma_tmp = np.zeros([3, 3])
		for j in range(0, n):
			sigma_tmp = sigma_tmp + gamma[j, i] * np.outer(data[j, 2:5] - miu_update[i], data[j, 2:5] - miu_update[i])
		sigma_upate[i] = sigma_tmp / N[i]

	# update pi
	pi_update = N / n 

	return miu_update, sigma_upate, pi_update


filename = sys.argv[1]
data, image = read_data(filename, False)

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
	if diff_norm < 0.01:

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


'''
# evaluate() function is very slow, therefore use the difference between parameter norm as stopping criteria
# initialize parameters
miu, sigma, pi = initialize(data)
# evaluate log likelihood
likelihood_orig = evaluate(data, miu, sigma, pi)
likelihood_update = likelihood_orig

count = 1
# EM iterations
while True:

	# evaluate responsibilities
	gamma = expectation(data, miu, sigma, pi)
	# re-estimate parameters
	miu, sigma, pi = maximization(data, gamma)

	likelihood_orig = likelihood_update
	# evaluate log likelihood
	likelihood_update = evaluate(data, miu, sigma, pi)

	# output runtime information
	print("Iteration " + str(count) + ": ")
	print(str(likelihood_update) + " - " + str(likelihood_orig) + " = " + str(likelihood_update - likelihood_orig) + "\n")
	count += 1
	# if the increase of the likelihood is smaller than 1, output the result and break EM iterations
	if likelihood_update - likelihood_orig < 1:
'''