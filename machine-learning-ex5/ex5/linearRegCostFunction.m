function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = 1/(2*m) * sum((X * theta - y).^2) + lambda/(2*m) * sum(theta(2:end,:).^2);



% had a very difficult time trying to vectorize this so I wouldn't have to 
% write two statements and make this only work for n = 2.  But the
% problem is not making the regularization params work in a single
% vectorized statement but instead the X(:,1) or X(:,2) term work
% in one statement. That's because you need to multiply the single
% value returned by X * theta - y by just one of the X columns, not both.
% So, finally bailed and did it with two lines
grad(1,1) = 1/m * sum((X * theta - y) .* X(:,1));
grad(2,1) = 1/m * sum((X * theta - y) .* X(:,2)) .+ ((lambda/m) * theta(2,1));
% here's the attempt at a single line, completely vectorized solution
% as is, the .* X won't work because X is 12 x 2, not 12 x 1
% grad = 1/m * sum((X * theta - y) .* X) .+ ((lambda/m) * [0; theta(2:end,:)]);


% =========================================================================

grad = grad(:);

end
