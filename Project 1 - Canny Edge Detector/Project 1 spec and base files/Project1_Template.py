#Do not import any additional modules
import numpy as np
from PIL.Image import open
import matplotlib.pyplot as plt

### Load, convert to grayscale, plot, and resave an image
I = np.array(open('Iribe.jpg').convert('L'))/255

plt.imshow(I,cmap='gray')
plt.axis('off')
plt.show()

plt.imsave('test.png',I,cmap='gray')

### Part 1
def gausskernel(sigma):
    #Create a 3*sigma x 3*sigma 2D Gaussian kernel
    h=np.array([[1.]])
    return h

def myfilter(I,h):
    #Appropriately pad I
    #Convolve I with h
    I_filtered=I
    return I_filtered

def check_thin(x, y, angles, magnitudes):
    """
    This returns True when the edge at (x,y) should be thinned 
    and False otherwise.

    Args:
        x: int, x coordinate of gradient value being examined
        y: int, y coordinate of gradient value being examined
        angles: array of values containing the binned angle at each image location
        magnitudes: array of values containing the gradient magnitude at each image location
    """
    pass


h1=np.array([[-1/9,-1/9,-1/9],[-1/9,2,-1/9],[-1/9,-1/9,-1/9]])
h2=np.array([[-1,3,-1]])
h3=np.array([[-1],[3],[-1]])


### Part 2
Sx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
Sy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
def myCanny(I,sigma=1,t_low=.5,t_high=1):
    #Smooth with gaussian kernel
    #Find img gradients
    #Thin edges
    #Hystersis thresholding
    from scipy.ndimage.measurements import label
    myedges=I
    return myedges

edges=myCanny(I,sigma=3,t_low=.5,t_high=1)
plt.imshow(edges, interpolation='none')
plt.show()