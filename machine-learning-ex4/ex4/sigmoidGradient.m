function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).


% sigmoid = 1/(1 + e^-z)
% the gradient is just the derivative which can 
% be calulated with:
% sigmoid * (1 - sigmoid)
% but z could be a vector or matrix
% 1 + matrix is an element wise operation
% but element-wise division is done with ./

sigmoid = 1.0 ./ (1.0 + exp(-z));
g = sigmoid .* (1 - sigmoid);












% =============================================================




end
