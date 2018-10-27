function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
        h = sigmoid(X * theta)
        J = 1/m * (-transpose(y) * log(h) - transpose(1 - y) * log(1 - h)) + lambda/(2*m) * sum(theta(2:end).^2);

%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%       ML: this is just like the exercise 2 regularized gradient function (just copied it here)
%           but don't you want a vector of gradients for each theta?  Yes, that's the point (
%           think about the non-vectorized version with a loop that calculates a gradient for each theta,
%           adding sum of lambda * thetas to only the non-zero theta gradients.
%           X is matrix of samples, h is column vector of predictions, y is column of answers (1 per sample) m x 1
%           so h-y is element-by operation and result is a single column vectorm m x 1
%           transpose(X) is an n x m matrix.  that * h-y is then a n x 1 matrix, 1 row per theta!!!
%           and that's what you want!!! a result of gradients, 1 per theta (that are subtracted from the corresponding
%           theta during gradient descent - but here we're just producing the gradients). Adding the regularization
%           parameter changes nothing about the dimensions of the vector.
%
           grad = 1/m * (transpose(X) * (h - y)) + lambda/m * [0; theta(2:end)];

%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable) lambda/m * theta, I think but is that applied to theta or sum(thetas)?  











% =============================================================

grad = grad(:);

end
