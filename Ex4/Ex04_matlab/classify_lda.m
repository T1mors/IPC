fuction vec_y_test = classify_lda(vec_w_opt_lda, vec_m1_train, vec_m2_train, mat_X_test);
% Compute the projection variable y for the test dataset, using the projection vector that you have learned on the training data.
% For classification, use the mid-point between the two class-means (vec_m1_train and vec_m2_train) as the decision threshold, i.e., map this mid-point to the output value zero in vec_y_test.
