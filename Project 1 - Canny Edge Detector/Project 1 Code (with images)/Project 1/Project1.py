#Do not import any additional modules
import numpy as np
from PIL.Image import open
import matplotlib.pyplot as plt
from scipy import ndimage

### Load, convert to grayscale, plot, and resave an image
I = np.array(open('Iribe.jpg').convert('L'))/255

### Part 1
def gausskernel(sigma):
    # Create a 3*sigma x 3*sigma 2D Gaussian kernel
    x, y = np.meshgrid(np.arange(-3*sigma, 3*sigma+1), np.arange(-3*sigma, 3*sigma+1))
    h = np.exp(-(x**2 + y**2) / (2*sigma**2)) / (2 * np.pi * sigma**2)
    h /= h.sum()
    return h


# Test the gausskernel function
sigmas = [1, 2, 3, 4, 5]
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10,3))
for i, sigma in enumerate(sigmas):
    h = gausskernel(sigma)
    axes[i].imshow(h, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title('Sigma = {}'.format(sigma))

plt.tight_layout()
plt.show()



def myImageFilter(I, h):
    # Get the dimensions of the input image and filter
    m, n = I.shape
    k, l = h.shape
    
    # Compute the amount of padding needed on each side
    pad_width = [(k // 2, k // 2), (l // 2, l // 2)]
    
    # Add zero-padding to the input image
    I_padded = np.pad(I, pad_width, mode='constant')
    
    # Initialize the output image
    I_filtered = np.zeros((m, n))
    
    # Perform convolution on the input image
    for i in range(m):
        for j in range(n):
            # Extract the patch of the input image that the filter overlaps
            patch = I_padded[i:i+k, j:j+l]
            
            # Compute the element-wise product of the patch and the filter
            patch_filtered = patch * h
            
            # Sum the element-wise products and save the result to the output image
            I_filtered[i, j] = np.sum(patch_filtered)
    
    return I_filtered

# Load the image
I = np.array(open('Iribe.jpg').convert('L'))/255

# Define the standard deviations for the Gaussian kernels
sigmas = [3, 5, 10]

# Apply the filters and plot the results
fig, axes = plt.subplots(nrows=1, ncols=len(sigmas)+2, figsize=(10,3))

# Plot the original image
axes[0].imshow(I, cmap='gray')
axes[0].axis('off')
axes[0].set_title('Original')

# Loop over the different standard deviations and apply the corresponding filter
for i, sigma in enumerate(sigmas):
    h = gausskernel(sigma)
    I_filtered = myImageFilter(I, h)
    axes[i+1].imshow(I_filtered, cmap='gray')
    axes[i+1].axis('off')
    axes[i+1].set_title('Sigma = {}'.format(sigma))

# Plot the Gaussian kernel with sigma = 10
h10 = gausskernel(10)
axes[-1].imshow(h10, cmap='gray')
axes[-1].axis('off')
axes[-1].set_title('Gaussian kernel\nwith sigma = 10')

plt.tight_layout()
plt.show()


I = np.array(open('Iribe.jpg').convert('L'))/255

# Apply the filters h1, h2, and h3 to the image

h1=np.array([[-1/9,-1/9,-1/9],[-1/9,2,-1/9],[-1/9,-1/9,-1/9]])
h2=np.array([[-1,3,-1]])
h3=np.array([[-1],[3],[-1]])

I_h1 = myImageFilter(I, h1)
I_h2 = myImageFilter(I, h2)
I_h3 = myImageFilter(I, h3)

# Display the filtered images
fig, axs = plt.subplots(1, 4, figsize=(10, 5))
axs[0].imshow(I, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(I_h1, cmap='gray')
axs[1].set_title('h1 filter')
axs[2].imshow(I_h2, cmap='gray')
axs[2].set_title('h2 filter')
axs[3].imshow(I_h3, cmap='gray')
axs[3].set_title('h3 filter')
for ax in axs:
    ax.axis('off')
plt.show()

### Part 2

def apply_hysteresis_thresholding(image, gradient_magnitude, t_low, t_high):
  # Threshold the gradient magnitude image
  strong_edges = (gradient_magnitude > t_high)
  no_edges = (gradient_magnitude < t_low)

  # Label connected components in the strong edges
  labels, num_labels = ndimage.label(strong_edges)

  # Find weak edges that are connected to strong edges
  for i in range(1, num_labels+1):
      # Find all pixels in the current connected component
      component = (labels == i)

      # Check if any weak edges are connected to this component
      if np.any(np.logical_and(component, ~no_edges)):
          strong_edges = np.logical_or(strong_edges, component)

  return strong_edges.astype(np.uint8)



I = np.array(open('Iribe.jpg').convert('L'))/255
Sx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
Sy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
def myCanny(I, sigma, t_low, t_high):
  fig, axs = plt.subplots(2, 4, figsize=(20, 10))
  axs[0][0].imshow(I)
  axs[0][0].set_title('Original')
  
  # Step 1: Apply a Gaussian filter to the image with standard deviation sigma
  h = gausskernel(sigma)
  I_smoothed = myImageFilter(I, h)
  axs[0][1].imshow(I_smoothed)
  axs[0][1].set_title('Step1')

  # Step 2: Compute the gradient magnitude and angle
  Ix = myImageFilter(I_smoothed, Sx)
  Iy = myImageFilter(I_smoothed, Sy)
  gradient_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
  gradient_angle = np.arctan2(Iy, Ix) * 180 / np.pi
  gradient_angle = (np.round(gradient_angle / 45) % 4 ) * 45 

  axs[0][2].imshow(Ix)
  axs[0][2].set_title('step2: IX')
  axs[0][3].imshow(Iy)
  axs[0][3].set_title('step2: IX')
  axs[1][0].imshow(gradient_magnitude)
  axs[1][0].set_title('step2: gradient_magnitude')
  axs[1][1].imshow(gradient_angle)
  axs[1][1].set_title('step2: gradient_angle')

  # Step 3: Edge thinning / Non-maximum suppression
  I_nms = np.zeros_like(gradient_magnitude)
  for i in range(1, I_nms.shape[0]-1):
      for j in range(1, I_nms.shape[1]-1):
          if gradient_angle[i, j] == 0:
              if (gradient_magnitude[i, j] >= gradient_magnitude[i, j-1]) and (gradient_magnitude[i, j] >= gradient_magnitude[i, j+1]):
                  I_nms[i, j] = gradient_magnitude[i, j]
          elif gradient_angle[i, j] == 45:
              if (gradient_magnitude[i, j] >= gradient_magnitude[i-1, j+1]) and (gradient_magnitude[i, j] >= gradient_magnitude[i+1, j-1]):
                  I_nms[i, j] = gradient_magnitude[i, j]
          elif gradient_angle[i, j] == 90:
              if (gradient_magnitude[i, j] >= gradient_magnitude[i-1, j]) and (gradient_magnitude[i, j] >= gradient_magnitude[i+1, j]):
                  I_nms[i, j] = gradient_magnitude[i, j]
          elif gradient_angle[i, j] == 135:
              if (gradient_magnitude[i, j] >= gradient_magnitude[i-1, j-1]) and (gradient_magnitude[i, j] >= gradient_magnitude[i+1, j+1]):
                  I_nms[i, j] = gradient_magnitude[i, j]
  
  axs[1][2].imshow(I_nms)
  axs[1][2].set_title('Step3: Edge thinning')

  # Step 4: Hysterisis thresholding
  final = apply_hysteresis_thresholding(I_nms, gradient_magnitude, t_low, t_high)

  axs[1][3].imshow(final)
  axs[1][3].set_title('Step4: Hysterisis thresholding')
  

#   for ax in axs:
#       ax.axis('off')
#plt.show()
    
  return final




    

edges=myCanny(I,sigma=1,t_low=0.4,t_high=0.8)
plt.imshow(edges, interpolation='none')
plt.show()


#Testing:

sigmas = [0.25, 0.5 ,1]
low = [0.2, 0.4, 0.6]
high = [0.6, 0.8, 1]

fig, axs = plt.subplots(3, 3, figsize=(20, 10))

for i in range(3):
  for j in range(3):
      edges=myCanny(I,sigma=sigmas[i] ,t_low=low[j],t_high=high[j])
      #plt.imshow(edges, interpolation='none')
      axs[i][j].imshow(edges)
      axs[i][j].set_title(f"Final image with: simga={sigmas[i]} t_low={low[j]} and t_high={high[j]}")

plt.show()

#Extra credit task:
# Read the two input images and make sure they have the same size
img1 = open('MM.jpeg').convert('L')
img2 = open('TC.jpeg').convert('L')
img2 = img2.resize(img1.size)

# Convert the PIL images to numpy arrays
img1 = np.array(img1)
img2 = np.array(img2)

# Compute the Fourier transforms of both images
f1 = np.fft.fft2(img1)
f2 = np.fft.fft2(img2)

# Shift the zero frequency component to the center of the spectrum
f1_shifted = np.fft.fftshift(f1)
f2_shifted = np.fft.fftshift(f2)

# Compute the magnitudes of the Fourier transforms
mag1 = np.abs(f1_shifted)
mag2 = np.abs(f2_shifted)

# A larger alpha gives more weight to the high frequencies of img2
alpha = 0.6

# Construct a hybrid spectrum by combining the low frequencies of img1 with the
# high frequencies of img2, using a weighted sum of the Fourier transforms
hybrid = alpha * f2_shifted + (1 - alpha) * f1_shifted

# Shift the hybrid spectrum back to the top-left corner of the spectrum
hybrid_shifted = np.fft.ifftshift(hybrid)

# Compute the inverse Fourier transform of the hybrid spectrum to obtain the hybrid image
hybrid_img = np.fft.ifft2(hybrid_shifted)

# Normalize the resulting image to the range [0, 255] and convert it to uint8 format
hybrid_img = np.abs(hybrid_img)
hybrid_img = (hybrid_img - hybrid_img.min()) / (hybrid_img.max() - hybrid_img.min()) * 255
hybrid_img = np.uint8(hybrid_img)

# Display the resulting hybrid image
plt.imshow(hybrid_img, cmap='gray')
plt.axis('off')
plt.show()