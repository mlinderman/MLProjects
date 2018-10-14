function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% feels a bit wrong to use a element-wise sigmoid function that returns a m x 1 vector
% (because X*theta is a m x 1 vector)
% and then take the log of that but it seems to work since log will perform a log operation
% on each element of the vector and then you multiply transposed y (1 x m) * h (m x 1)
% which results in a scalar value

h = sigmoid(X*theta); % X*theta aligns the feature values * thetas for each feature, resulting in m * 1 vector
J = 1/m * (-transpose(y) * log(h) - transpose(1 - y) * log(1 - h));

% calculate the gradient for this theta ( I guess that's just the theta without iterating and subtracting using an alpha multiplier)

grad = 1/m * (transpose(X) * (h - y));


% =============================================================

end
