function [mat_features, NCol, NRow, NFrame] = matrix2vector(mda_data)
% Convert each image matrix in mda_data into a column vector.
% Store the column vectors in matrix mat_feature.

NCol   = size(mda_data,1);
NRow   = size(mda_data,2);
NFrame = size(mda_data,3);

% rearrange data into data matrix mat_features
** your code here **
