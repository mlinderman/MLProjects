function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% you want num_labels results for each X, so get the matrices lined up 
% so that the result is a matrix of n rows x num_labels columns
% then apply max() to each row (all rows should total 1 because only one classifier
% should match) and get the returned index from that max operation
% that index of the max column should be the column corresponding to the classifier, 1-10
% Geet max to operate along columns, not rows (3rd argument to max())
% max returns max found along the searched dimension (columns in this case), 
% and, optionally, the index of the column which should be the only column 
% for that row containing 1 since it's presumed that only one qualifier matches.
[max, p] = max(X * transpose(all_theta), [], 2) 


% =========================================================================


end
