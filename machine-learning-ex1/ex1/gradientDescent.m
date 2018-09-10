function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %

    % ML: doing this in a generic loop for all theta values so that it will
    % work for more than just 2 features (the theta vector has only 2 values in this example since 
    % the hypothisis is h(x) = theta0*X0 + theta1*X1  )
    % note: the X[:,idx] in the formula is the X column that corresponds to the theta row currently 
    % being processed.
    % So you''re multiplying a vector (X*theta - y) and a vector (X[:,idx]) - selecting the column
    % that corresponds to the theta being calculated and doing an element-wise multiplication
    % so if X = [1,2; 3,4; 5,6] theta (a column vector as defined in ex1.m) could be [1;2]
    % and to multiply by the right x you need just that column so X[:,idx] is all rows, one column
    % One other thing: "simulaneous update": newThetas variable is necessary to record newly calculated 
    % theta values while iterating.  If you don''t keep them separate, theta will be changing during 
    % an iteration.
    % My results were very close to expected without that but still wrong so it does make a difference
    newThetas = zeros([size(theta), 1]);
    for idx = 1:size(theta) % size of the largest dimension, the row count, in this instance
        newThetas(idx, 1) = theta(idx) - alpha * 1/m * sum((X * theta - y) .* X(:, idx));
    end
    theta = [newThetas];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
