function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;
% after running the code below, found that optimal values of C, sigma
% (that resulted in the lowest % of misses against the validation set)
% were C = 1; sigma = 0.1.  Commented out starter values above in favor of these
C = 1;
sigma = 0.1;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

% that ~= operator is not equals.  So, you're taking the mean 
% value of all the predictions but first converting them to 0 or 1
% so, you'll get a percentage of predictions that is incorrect
% if 10 examples and you correctly predict 8 of them, 
% the total values of ~= comparisons is 2.  That over 10 is .2 or 20%
%

%{
testValues = [.01, .03, 0.1, 0.3, 1, 3, 10, 30];
predictResults = [0, 3];
x1 = X(:,1);
x2 = X(:,2);
predictCount = 0;

for C = [testValues]
    for sigma = [testValues]
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        predictResults(++predictCount, 1:3) = [C, sigma, mean(double(predictions ~= yval))];
    end;
end;

predictResults
[min, idx] = min(predictResults(:,3))
C = predictResults(idx,1)
sigma = predictResults(idx,2)
%}


% =========================================================================

end
