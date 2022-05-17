import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from functions import *
# main script for exercise 4
# Information Processing and Communication Lecture
# May 2022
# Jörn Anemüller

# Selection of two (out of the ten) digits that you will use to train a two-class discriminative classifier

DigitA = 3 # just chosen as an example, pick your own
DigitB = 8

# Loading the MNIST digits dataset
# train_images and test_images contain gray-scale images of size 28x28 pixels
#   e.g., size(train_images) : (28, 28, 60000)
# train_labels and test_labes contain the corresponding class index for each image
#    e.g., size(train_labels) : (60000, 1)

train_images,train_labels,test_images,test_labels = mnistRead()
train_images = float(train_images)
train_labels = float(train_labels)
test_images = float(test_images)
test_labels = float(test_labels)


# Plot a number of example images to see how the data look like.
vPlotInd = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # % insert 12 indices here

for Ind in np.arange(1,len(vPlotInd)+1).reshape(-1):
    plt(3,4,vPlotInd(Ind))
    ax.imshow(train_images,vPlotInd(Ind))
    plt.axis('off')
    plt.title(train_labels(vPlotInd(Ind)))

# Converting data matrix to feature vector
# Convert each 2-dimensional image matrix in train_images and test_images into a feature vector that is used for optimization.
# Simple reshaping of the data does the job, and should be performed for each image stored in the multi-dimensional data.

mat_features_train,NCol,NRow,NFrame = matrix2vector(train_images)
mat_features_test,NCol,NRow,NFrame = matrix2vector(test_images)

'''
find all occurences of DigitA and DigitB in the label vector
fill in the gaps indicated by the dots ...
alternatively, this can also be done by logical indexing in matlab
[...vIndA_train...] = find(...train_labels...);
[...vIndB_train...] = find(...train_labels...);
[...vIndA_test...]  = find(...test_labels...);
[...vIndB_test...]  = find(...test_labels...);
'''

# extract the corresponding columns from mat_features_train and mat_features_test to form train matrix mat_X and test matrix mat_X_test
mat_X = mat_features_train()
mat_X_test = mat_features_test()

# form the correct two-class label vector vec_y_true and vec_y_true_test corresponding to mat_X and mat_X_test
vec_y_true = []
vec_y_true_test = []

# perform any additional preprocessing of the data the you might find is necessary
#...
# hint: there is one additional step that is necessary

#
#  Classification with the least-squares approach  #
#
vec_y = []
# Find the optimum weight vector and return also the projection variable vec_y for the LDA method
vec_w_opt_lda,vec_y_lda = learn_lda(mat_X, vec_y)

# compute the accuracy on the training data, i.e., number of correctly classified images divided by total number of images
accuracy_train_lda = compute_accuracy(vec_y_true, vec_y)
vec_w_opt = []
# Reshape the optimum weight vector to matrix
mat_weights = vector2matrix(vec_w_opt)

'''
Visualize the weights matrix
Plot the weight matrix as an image in order to visualize extracted information about the optimal discriminative surface
** your code here **
'''
vec_m1_train = []
vec_m2_train = []
# compute the decision variable vec_y_hat_test for the test dataset, using the classification model that you have learnt on the training data
vec_y_test = classify_lda(vec_w_opt_lda,vec_m1_train,vec_m2_train,mat_X_test)

# compute the accuracty on the test data, i.e., the portion of the data that has not been used for training
accuracy_test_lda = compute_accuracy(vec_y_true_test,vec_y_test)

# plot some arbitrarily drawn examples from the test dataset, and indicate that class label that the classifier assigned to each of them