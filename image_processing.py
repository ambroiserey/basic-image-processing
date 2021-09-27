from scipy import ndimage
from skimage.segmentation import slic
from skimage.color import rgb2gray
import skimage.feature
import skimage.viewer
from skimage.transform import probabilistic_hough_line
import skimage.filters
import matplotlib.pyplot as plt
import numpy as np

# Read avengers_imdb.jpg
avengers = skimage.io.imread("avengers_imdb.jpg")

print("Size of the avengers_imdb.jpg image:", avengers.shape)

# Convert the image to grayscale
avengers_grayscale = rgb2gray(avengers)

# Convert the image to black and white
thresh = skimage.filters.threshold_otsu(avengers_grayscale)
avengers_black_and_white = avengers_grayscale > thresh

# Show the grayscale image
plt.imshow(avengers_grayscale, cmap="gray")
plt.title("Grayscale image")
plt.show()

# Show the black and white image
plt.imshow(avengers_black_and_white, cmap="gray")
plt.title("Black and white image")
plt.show()

# Read bush_house_wikipedia.jpg
bush_house_wikipedia = skimage.io.imread("bush_house_wikipedia.jpg")

# Add Gaussian random noise to the image with variance = 0.1
noisy_bush_house_wikipedia = skimage.util.random_noise(bush_house_wikipedia, mode="gaussian", seed=None, clip=True, var=0.1)

# Show the image with the Gaussian noise
plt.imshow(noisy_bush_house_wikipedia)
plt.title("Gaussian random noise")
plt.show()

# Filter the image with a Gaussian mask with sigma = 1
gaussian_filter = ndimage.gaussian_filter(noisy_bush_house_wikipedia, sigma = 1)

# Show the image with the Gaussian filter
plt.imshow(gaussian_filter)
plt.title("Gaussian filter")
plt.show()

# Filter the image with a uniform mask of size 9x9
uniform_filter = ndimage.uniform_filter(gaussian_filter, size = 9)

# Show the image with the uniform filter
plt.imshow(uniform_filter)
plt.title("Uniform filter")
plt.show()

# Read forestry_commission_gov_uk.jpg
forestry_commission_gov_uk = skimage.io.imread("forestry_commission_gov_uk.jpg")

# Divide the image into 5 segments using k-means segmentation
segments = slic(forestry_commission_gov_uk, n_segments = 5, compactness = 25)

# Show the image with the 5 segments
plt.imshow(segments)
plt.title("Segmented image")
plt.show()

# I took some code and adapted it from : https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html

# Read rolland_garros_tv5monde.jpg and convert it to gray
rolland_garros_tv5monde_gray = skimage.io.imread("rolland_garros_tv5monde.jpg", as_gray=True)

# Perform Canny edge detection on rolland_garros_tv5monde.jpg
edges = skimage.feature.canny(image = rolland_garros_tv5monde_gray, sigma = 1, low_threshold = 0.3, high_threshold = 0.6)

# Show the image with the Canny edge detection
plt.imshow(edges)
plt.title("Canny edge detection")
plt.show()

# Apply Hough probabilistic transformation on the image with Canny edge detection
lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

plt.imshow(edges * 0)

for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

# Show the image with the Hough probabilistic transformation
plt.title("Hough probabilistic transformation")
plt.show()
