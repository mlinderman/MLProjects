function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    newThetas = zeros([size(theta), 1]);
    for idx = 1:size(theta) % size of the largest dimension, the row count, in this instance
        % using X*theta, not theta'X because X samples are rows, not columns and to do this vector/matrix 
        % multiplication, we want the features for each row of X to be multiplied to the corresponding thetas.'
        newThetas(idx, 1) = theta(idx) - alpha * 1/m * sum((X * theta - y) .* X(:, idx));
    end

    theta = [newThetas];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

