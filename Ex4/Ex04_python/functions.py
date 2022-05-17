
def classify_lda(vec_w_opt_lda = None,vec_m1_train = None,vec_m2_train = None,mat_X_test = None): 
    # Compute the projection variable y for the test dataset, using the projection vector that you have learned on the training data.
# For classification, use the mid-point between the two class-means (vec_m1_train and vec_m2_train) as the decision threshold, i.e., map this mid-point to the output value zero in vec_y_test.
    return vec_y_test

def compute_accuracy(vec_y_true = None,vec_y = None): 
    # compute the accuracy on the training data, i.e., number of correctly classified images divided by total number of images
    return accuracy_train_lda

def learn_lda(mat_X = None,vec_y_true = None): 
    # Find the optimum weight vector and return also the projection variable vec_y_lda for the LDA method
    return vec_w_opt_lda,vec_m1_train,vec_m2_train,vec_y_lda

def matrix2vector(mda_data = None): 
    # Convert each image matrix in mda_data into a column vector.
# Store the column vectors in matrix mat_feature.
    
    NCol = mda_data.shape[1-1]
    NRow = mda_data.shape[2-1]
    NFrame = mda_data.shape[3-1]
    # rearrange data into data matrix mat_features
    return mat_features,NCol,NRow,NFrame

import numpy as np
    
def mnistRead(): 
    # mnistRead Read in MNIST digit set in Le Cun's format
# function [trainImages, trainLabels, testImages, testLabels] = readMNIST()
    
    # The data is available at
# http://yann.lecun.com/exdb/mnist/
    
    # OUTPUT:
# trainImages(:,:,i) is a uint8 matrix of size 28x28x60000
#         0 = background, 255 = foreground
# trainLabels(i) - 60000x1 uint8 vector
# testImages(:,:,i) size 28x28x10,000
# testLabels(i)
    
    # Use mnistShow(trainImages, trainLabels) to visualize data.
    
    # This function was originally written by Bryan Russell
# www.ai.mit.edu/~brussell
# Modified by Kevin Murphy, 9 February 2004
# www.ai.mit.edu/~murphyk
    
    fid = open('train-images-idx3-ubyte','r','ieee-be')
    
    A = fread(fid,4,'uint32')
    num_images = A(2)
    mdim = A(3)
    ndim = A(4)
    train_images = fread(fid,mdim * ndim * num_images,'uint8=>uint8')
    train_images = np.reshape(train_images, tuple(np.array([mdim,ndim,num_images])), order="F")
    train_images = permute(train_images,np.array([2,1,3]))
    fid.close()
    fid = open('train-labels-idx1-ubyte','r','ieee-be')
    A = fread(fid,2,'uint32')
    num_images = A(2)
    train_labels = fread(fid,num_images,'uint8=>uint8')
    fid.close()
    # Test
    
    fid = open('t10k-images-idx3-ubyte','r','ieee-be')
    A = fread(fid,4,'uint32')
    num_images = A(2)
    mdim = A(3)
    ndim = A(4)
    test_images = fread(fid,mdim * ndim * num_images,'uint8=>uint8')
    test_images = np.reshape(test_images, tuple(np.array([mdim,ndim,num_images])), order="F")
    test_images = permute(test_images,np.array([2,1,3]))
    fid.close()
    # Testing labels:
    fid = open('t10k-labels-idx1-ubyte','r','ieee-be')
    A = fread(fid,2,'uint32')
    num_images = A(2)
    test_labels = fread(fid,num_images,'uint8=>uint8')
    fid.close()
    return train_images,train_labels,test_images,test_labels

import numpy as np
import matplotlib.pyplot as plt
    
def mnistShow(images = None,labels = None): 
    # showMNIST Display MNIST digits; press button inside window to step through
# function showMNIST(images, labels)
    
    if len(varargin) < 2:
        labels = []
    
    for k in np.arange(1,images.shape[3-1]+1).reshape(-1):
        image(images(:,:,k))
        colormap('gray')
        plt.axis('image','off')
        if not len(labels)==0 :
            plt.title(num2str(float(labels(k))))
        waitforbuttonpress

def vector2matrix(vec_in = None,NCol = None,NRow = None): 
    # Convert the parameter vector vec_in into a matrix mat_out
# The size of mat_out is (NRow, NCol)
    return mat_out