import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from numpy.linalg import det
import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import shift
from scipy.ndimage import rotate

###### LOAD IMAGE ######################################################

# Load the image
im1 = Image.open("CircleLineRect.png")
im2 = Image.open("zurlim.png")



######  Harris Corner Detector ######################################################

def Harrison_Corner(im, sigma = 1 , threshold = 1):

    # Convert the image to grayscale matrix
    im_gray = im.convert('L')

    # Convert the grayscale image to a NumPy array
    im_array = np.array(im_gray)

    # Test ROTATE UNIFORME AND SHIFT
    #im_array = uniform_filter(im_array)
    #im_array = rotate(im_array, 20)
    #im_array = shift(im_array, 20)

    # Convolution Sobel filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Perform convolution
    ix = convolve2d(im_array, sobel_x, mode='same')
    iy = convolve2d(im_array, sobel_y, mode='same')

    # Intermediate results
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(ix, cmap='gray')
    plt.title('Convolution with Sobel X')
    plt.subplot(1, 2, 2)
    plt.imshow(iy, cmap='gray')
    plt.title('Convolution with Sobel Y')
    plt.show()
    '''

    # Do the product between 2 images
    def produit_static(image1, image2):
        out_image = np.zeros_like(image1)
        largeur, hauteur = ix.shape
        for x in range(largeur):
            for y in range(hauteur):
                pixel1 = image1[(x, y)]
                pixel2 = image2[(x, y)]
                out_image[x, y]= (pixel1 * pixel2)
        return(out_image)

    ix2 = produit_static(ix, ix)
    iy2 = produit_static(iy, iy)
    ixy = produit_static(ix, iy)

    # Intermediate results
    '''
    plt.subplot(1, 3, 1)
    plt.imshow(ix2, cmap='gray')
    plt.title('Ixx')
    plt.subplot(1, 3, 2)
    plt.imshow(iy2, cmap='gray')
    plt.title('Iyy')
    plt.subplot(1, 3, 3)
    plt.imshow(ixy, cmap='gray')
    plt.title('Ixy')
    plt.show()
    '''

    ix2_smooth = gaussian_filter(ix2, sigma=sigma)
    iy2_smooth = gaussian_filter(iy2, sigma=sigma)
    ixy_smooth = gaussian_filter(ixy, sigma=sigma)

    # Intermediate results
    '''
    plt.subplot(1, 3, 1)
    plt.imshow(ix2_smooth, cmap='gray')
    plt.title('Ixx smooth')
    plt.subplot(1, 3, 2)
    plt.imshow(iy2_smooth, cmap='gray')
    plt.title('Iyy smooth')
    plt.subplot(1, 3, 3)
    plt.imshow(ixy_smooth, cmap='gray')
    plt.title('Ixy smooth')
    plt.show()
    '''

    # Create A and H
    h, w = im.size
    A = np.zeros((h, w, 2, 2))
    H = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            A[i][j] = np.array([[ix2_smooth[i][j], ixy_smooth[i][j]], [ixy_smooth[i][j], iy2_smooth[i][j]]])
            H[i][j] = det(A[i][j]) - 0.05 * np.trace(A[i][j]) ** 2
    
    # Intermediate results
    '''
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow((H), cmap='gray')
    plt.title('H matrix')
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.title('Original Image')
    plt.show()
    '''

    # Apply non max
    harrison_matrix = np.zeros_like(H)
    for i in range(len(harrison_matrix)-1):
        for j in range(len(harrison_matrix[0])-1):
            voisinage = [H[i, j], H[i+1, j], H[i-1, j], H[i, j+1], H[i, j-1], H[i+1, j+1], H[i-1, j-1], H[i+1, j-1], H[i-1, j+1]]
            if max(voisinage) == H[i, j] and H[i, j] > threshold:
                harrison_matrix[i, j] = 255


    # Print Results
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow((harrison_matrix), cmap='gray')
    plt.title('Harrison Corner Detection')
    plt.subplot(1, 2, 1)
    plt.imshow(im_array, cmap='gray')
    plt.title('Original Image')
    plt.show()

    return harrison_matrix



######  Canny Edge Detection ######################################################


def Canny_Edge_Dectection(im, sigma = 1, threshold = 1, error = 0):

    # Convert the image to grayscale
    im_gray = im.convert('L')

    # Convert the grayscale image to a NumPy array
    im_array = np.array(im_gray)

    # Smooth the image with gaussian filter
    im_smooth = gaussian_filter(im_array, sigma=sigma)

    # Intermediate results
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(im_smooth, cmap='gray')
    plt.title('Smooth image with Gaussian filter')
    plt.subplot(1, 2, 2)
    plt.imshow(im_array, cmap='gray')
    plt.title('Real image with Gaussian filter')
    plt.show()
    '''

    # Convolution Sobel filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Perform convolution
    ix = convolve2d(im_smooth, sobel_x, mode='same', boundary='wrap')
    iy = convolve2d(im_smooth, sobel_y, mode='same', boundary='wrap')

    # Intermediate results
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(ix, cmap='gray')
    plt.title('Convolution with Sobel X')
    plt.subplot(1, 2, 2)
    plt.imshow(iy, cmap='gray')
    plt.title('Convolution with Sobel Y')
    plt.show()
    '''

    # Find direction and magnitude in each point
    direction = np.arctan2(iy, ix) # Handles 0 divisions 
    magnitude = np.hypot(ix, iy)

    # Intermediate results

    plt.subplot(1, 2, 1)
    plt.imshow(direction, cmap='gray')
    plt.title('Direction')
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Magnitude')
    plt.show()

    edges_matrix = np.zeros_like(ix)

    pi = np.pi
    for i in range(len(edges_matrix)-1):
        for j in range(len(edges_matrix[0])-1):

            angle = direction[i, j]
            voisinage = []

            # Cas 1
            if ((angle < pi/2 + pi/8) and (angle > pi/2 - pi/8)) or ((angle < -pi/2 + pi/8) and (angle > -pi/2 - pi/8)) :
                voisinage = [magnitude[i, j-1], magnitude[i, j+1]]

            # Cas 2
            if ((angle < pi/4 + pi/8) and (angle > pi/4 - pi/8)) or ((angle < -3*pi/4 + pi/8) and (angle > -3*pi/4 - pi/8)) :
                voisinage = [magnitude[i-1, j+1], magnitude[i+1, j-1]]

            # Cas 3
            if ((angle < pi/8) and (angle > - pi/8)) or (angle < -pi + pi/8) or (angle > pi - pi/8) :
                voisinage = [magnitude[i+1, j], magnitude[i-1, j]]

            # Cas 4
            if ((angle < -pi/4 + pi/8) and (angle > -pi/4 - pi/8)) or ((angle < 3*pi/4 + pi/8) and (angle > 3*pi/4 - pi/8)) :
                voisinage = [magnitude[i+1, j+1], magnitude[i-1, j-1]]

            if magnitude[i, j] >= max(voisinage) - error and magnitude[i, j] > threshold :
                edges_matrix[i, j] = 255

    plt.subplot(1, 2, 1)
    plt.imshow(edges_matrix, cmap='gray')
    plt.title('Edge detection')
    plt.subplot(1, 2, 2)
    plt.imshow(im_array, cmap='gray')
    plt.title('Real image')
    plt.show()

def Canny_Edge_Dectection_error(im, sigma = 1, threshold = 1, error = 0):

    # Convert the image to grayscale
    im_gray = im.convert('L')

    # Convert the grayscale image to a NumPy array
    im_array = np.array(im_gray)

    # Smooth the image with gaussian filter
    im_smooth = gaussian_filter(im_array, sigma=sigma)

    # Intermediate results
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(im_smooth, cmap='gray')
    plt.title('Smooth image with Gaussian filter')
    plt.subplot(1, 2, 2)
    plt.imshow(im_array, cmap='gray')
    plt.title('Real image with Gaussian filter')
    plt.show()
    '''

    # Convolution Sobel filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Perform convolution
    ix = convolve2d(im_smooth, sobel_x, mode='same', boundary='wrap')
    iy = convolve2d(im_smooth, sobel_y, mode='same', boundary='wrap')

    # Intermediate results
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(ix, cmap='gray')
    plt.title('Convolution with Sobel X')
    plt.subplot(1, 2, 2)
    plt.imshow(iy, cmap='gray')
    plt.title('Convolution with Sobel Y')
    plt.show()
    '''

    # Find direction and magnitude in each point
    direction = np.arctan2(iy, ix) # Handles 0 divisions 
    magnitude = np.hypot(ix, iy)

    # Intermediate results
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(direction, cmap='gray')
    plt.title('Direction')
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Magnitude')
    plt.show()
    '''
    edges_matrix = np.zeros_like(ix)

    pi = np.pi
    for i in range(len(edges_matrix)-1):
        for j in range(len(edges_matrix[0])-1):

            angle = direction[i, j]
            voisinage = []

            # Cas 1
            if ((angle < pi/2 + pi/8) and (angle > pi/2 - pi/8)) or ((angle < -pi/2 + pi/8) and (angle > -pi/2 - pi/8)) :
                voisinage = [magnitude[i, j-1], magnitude[i, j+1]]

            # Cas 2
            if ((angle < pi/4 + pi/8) and (angle > pi/4 - pi/8)) or ((angle < -3*pi/4 + pi/8) and (angle > -3*pi/4 - pi/8)) :
                voisinage = [magnitude[i-1, j+1], magnitude[i+1, j-1]]

            # Cas 3
            if ((angle < pi/8) and (angle > - pi/8)) or (angle < -pi + pi/8) or (angle > pi - pi/8) :
                voisinage = [magnitude[i+1, j], magnitude[i-1, j]]

            # Cas 4
            if ((angle < -pi/4 + pi/8) and (angle > -pi/4 - pi/8)) or ((angle < 3*pi/4 + pi/8) and (angle > 3*pi/4 - pi/8)) :
                voisinage = [magnitude[i+1, j+1], magnitude[i-1, j-1]]

            if magnitude[i, j] >= max(voisinage) - error and magnitude[i, j] > threshold :
                edges_matrix[i, j] = 255

    plt.subplot(1, 2, 1)
    plt.imshow(edges_matrix, cmap='gray')
    plt.title('Edge detection allowed edge errors')
    plt.subplot(1, 2, 2)
    plt.imshow(im_array, cmap='gray')
    plt.title('Real image')
    plt.show()

Harrison_Corner(im1, 1, 150)
Harrison_Corner(im2, 1, 200)
Canny_Edge_Dectection(im1, 0.5, 150)
Canny_Edge_Dectection(im2, 0.5, 150)
#Canny_Edge_Dectection_error(im1, 0.5, 150, 30)
#Canny_Edge_Dectection_error(im2, 0.5, 150, 30)
