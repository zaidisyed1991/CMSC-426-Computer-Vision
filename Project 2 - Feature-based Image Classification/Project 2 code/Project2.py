import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.decomposition import PCA



# Define the paths to the training images
pathTrain = "Project2_data/TrainingDataset/"
pathTest = "Project2_data/TestingDataset/"

# Initialize the SIFT feature extractor
sift = cv2.SIFT_create()

# Initialize dictionaries to store features and descriptors for each class
butterfly_features = {}
cowboy_hat_features = {}
airplane_features = {}

# Loop over the images in each class and extract SIFT features
for filename in os.listdir(pathTrain):
  img = cv2.imread(pathTrain + filename)
  if "024" in filename:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    butterfly_features[filename] = (kp, des)

  if "051" in filename:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    cowboy_hat_features[filename] = (kp, des)

  if "251" in filename:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    airplane_features[filename] = (kp, des)

from sklearn.cluster import KMeans

# Concatenate the descriptors from all classes into a single array
descriptors = []
for _, des in butterfly_features.values():
    descriptors += des.tolist()
for _, des in cowboy_hat_features.values():
    descriptors += des.tolist()
for _, des in airplane_features.values():
    descriptors += des.tolist()

descriptors = np.array(descriptors)

# Cluster the descriptors using k-means clustering
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=100)
kmeans.fit(descriptors)

# Save the cluster centers as the visual words
visual_words = kmeans.cluster_centers_

# Initialize dictionaries to store histograms for each class
butterfly_hist = {}
cowboy_hat_hist = {}
airplane_hist = {}

# Loop over the images in each class and form histograms of N bins
for filename, (kp, des) in butterfly_features.items():
    histogram = np.zeros(n_clusters)
    for d in des:
        dist = np.linalg.norm(visual_words - d, axis=1)
        idx = np.argmin(dist)
        histogram[idx] += 1
    butterfly_hist[filename] = histogram

for filename, (kp, des) in cowboy_hat_features.items():
    histogram = np.zeros(n_clusters)
    for d in des:
        dist = np.linalg.norm(visual_words - d, axis=1)
        idx = np.argmin(dist)
        histogram[idx] += 1
    cowboy_hat_hist[filename] = histogram

for filename, (kp, des) in airplane_features.items():
    histogram = np.zeros(n_clusters)
    for d in des:
        dist = np.linalg.norm(visual_words - d, axis=1)
        idx = np.argmin(dist)
        histogram[idx] += 1
    airplane_hist[filename] = histogram


# Normalize the bin counts by dividing by the total number of SIFT features binned
for hist in butterfly_hist.values():
    hist /= np.sum(hist)
for hist in cowboy_hat_hist.values():
    hist /= np.sum(hist)
for hist in airplane_hist.values():
    hist /= np.sum(hist)

# Visualize a few histograms
fig, axs = plt.subplots(1, 3, figsize=(15, 3))
axs[0].bar(range(n_clusters), butterfly_hist['024_0001.jpg'])
axs[0].set_title('Butterfly')
axs[0].set_xlabel('Cluster')
axs[0].set_ylabel('Normalized Count')
axs[1].bar(range(n_clusters), cowboy_hat_hist['051_0013.jpg'])
axs[1].set_title('Cowboy Hat')
axs[1].set_xlabel('Cluster')
axs[1].set_ylabel('Normalized Count')
axs[2].bar(range(n_clusters), airplane_hist['251_0050.jpg'])
axs[2].set_title('Airplane')
axs[2].set_xlabel('Cluster')
axs[2].set_ylabel('Normalized Count')
plt.savefig("histograms.pdf")
plt.show()

# Find SIFT feature descriptors within each test image
Train_sift = {}
for filename in os.listdir(pathTest):
  img = cv2.imread(pathTest + filename)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  kp, des = sift.detectAndCompute(gray, None)
  Train_sift[filename] = (kp, des)


Train_hist = {}

for filename, (kp, des) in Train_sift.items():
    histogram = np.zeros(n_clusters)
    for d in des:
        dist = np.linalg.norm(visual_words - d, axis=1)
        idx = np.argmin(dist)
        histogram[idx] += 1
    histogram /= np.sum(histogram)
    Train_hist[filename] = histogram

# Find SIFT feature descriptors within each test image
test_sift = {}
for filename in os.listdir(pathTest):
  img = cv2.imread(pathTest + filename)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  kp, des = sift.detectAndCompute(gray, None)
  test_sift[filename] = (kp, des)


test_hist = {}

for filename, (kp, des) in test_sift.items():
    histogram = np.zeros(n_clusters)
    for d in des:
        dist = np.linalg.norm(visual_words - d, axis=1)
        idx = np.argmin(dist)
        histogram[idx] += 1
    histogram /= np.sum(histogram)
    test_hist[filename] = histogram


train_features, train_labels = [], []

for filename, value in Train_hist.items():
  train_features.append(value)
  if "024_" in filename:
    train_labels.append(0)
  if "051_" in filename:
    train_labels.append(1)
  if "251_" in filename:
    train_labels.append(2)

test_features, test_labels = [], []
numberOfbutterflies ,numberOfHats, numberOfAirPlanes = 0, 0 ,0

for filename, value in test_hist.items():
  test_features.append(value)
  if "024_" in filename:
    test_labels.append(0)
    numberOfbutterflies += 1 
  if "051_" in filename:
    test_labels.append(1)
    numberOfHats += 1 
  if "251_" in filename:
    test_labels.append(2)
    numberOfAirPlanes += 1


clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(train_features, train_labels)

# Predict the labels of the test images using the trained classifier
predicted_labels = clf.predict(test_features)

# Compute the fraction of the test set that was correctly classified
accuracy = np.mean(predicted_labels == test_labels)
print("Accuracy for K=1 Nearest Neighbor: {:.2%}".format(accuracy))

res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


for i in range(len(test_labels)):
  res[test_labels[i]][predicted_labels[i]] +=1

for i in range(3):
  res[0][i] =  (res[0][i]/numberOfbutterflies) * 100
for i in range(3):
  res[1][i] =  (res[1][i]/numberOfHats) * 100
for i in range(3):
  res[2][i] =  (res[2][i]/numberOfAirPlanes) * 100

""" Linear Support Vector Machine"""



clf = svm.LinearSVC()
clf.fit(train_features, train_labels)

# Predict the labels of the test images using the trained classifier
predicted_labels = clf.predict(test_features)

# Compute the fraction of the test set that was correctly classified
accuracy = np.mean(predicted_labels == test_labels)
print("Accuracy of Linear SVM: {:.2%}".format(accuracy))

res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


for i in range(len(test_labels)):
  res[test_labels[i]][predicted_labels[i]] +=1

for i in range(3):
  res[0][i] =  (res[0][i]/numberOfbutterflies) * 100
for i in range(3):
  res[1][i] =  (res[1][i]/numberOfHats) * 100
for i in range(3):
  res[2][i] =  (res[2][i]/numberOfAirPlanes) * 100

from sklearn.svm import SVC

# create a kernel SVM classifier with a radial basis function kernel
clf = SVC(kernel='rbf')

# fit the classifier on the training data
clf.fit(train_features, train_labels)

# predict the labels for the test data
pred_labels = clf.predict(test_features)

# calculate the accuracy
accuracy = np.mean(pred_labels == test_labels)
print("Accuracy of kernel SVM: {:.2f}%".format(accuracy*100))

res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


for i in range(len(test_labels)):
  res[test_labels[i]][pred_labels[i]] +=1

for i in range(3):
  res[0][i] =  (res[0][i]/numberOfbutterflies) * 100
for i in range(3):
  res[1][i] =  (res[1][i]/numberOfHats) * 100
for i in range(3):
  res[2][i] =  (res[2][i]/numberOfAirPlanes) * 100




# Define the colors for each class
color_mapping = {
    "butterfly": "red",
    "cowboy hat": "green",
    "airplane": "blue"
}

mapped = []


# Combine the histograms into one dictionary and add a label for each class
histograms = {}
labels = {}

for k, v in butterfly_hist.items():
    histograms[k] = v
    labels[k] = "butterfly"
    
for k, v in cowboy_hat_hist.items():
    histograms[k] = v
    labels[k] = "cowboy hat"
    
for k, v in airplane_hist.items():
    histograms[k] = v
    labels[k] = "airplane"

# Convert the histograms to a numpy array
X = np.array(list(histograms.values()))

# Compute the first three principal components of the combined data using PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create a scatter plot of the projected data with different colors for each class
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

for i, (k, v) in enumerate(histograms.items()):
    label = labels[k]
    color = color_mapping[label]

    x = X_pca[i, 0]
    y = X_pca[i, 1]
    z = X_pca[i, 2]
    if color in mapped:
      ax.scatter(x, y, z, c=color,  marker='o')
    else:
      ax.scatter(x, y, z, c=color,  marker='o', label=label)
      mapped.append(color)
      

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()
plt.savefig("Visualizing.pdf")
plt.show()