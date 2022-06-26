% load data from original file
load spambase.data

% we are interested in 
% -- the word occurence (0/1) -> columns 1:48
% -- the class-labels (0=not spam / 1=spam) -> column 58
vecPosData  = [1:48];
vecPosLabel = 58;

% at the same time, subdivide data into 
% -- 80% training data (rows 1:1450 and rows 1814:4042)
% -- 20% test data (rows 1451:1813 and rows 4043:4601)
vecPosTrain = [   1:1450, 1814:4042];
vecPosTest  = [1451:1813, 4043:4601];

% extract word occurence data and transform to 0/1, noting that sign(x=0) = 0
% and sign(x>0) = 1
matDataTrain  = sign(spambase(vecPosTrain, vecPosData));
matDataTest   = sign(spambase(vecPosTest,  vecPosData));

% extract class labels
vecLabelTrain = spambase(vecPosTrain, vecPosLabel);
vecLabelTest  = spambase(vecPosTest,  vecPosLabel);

% clean-up original data variable
%clear spambase

%%%   your algorithm follows below




