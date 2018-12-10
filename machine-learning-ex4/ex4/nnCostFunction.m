function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add bias column of 1's to X and call it a1
% a1 = 5000 x 401
% Theta1 = 25 x 401
% Theta2 = 10 x 26

% Note: multiplication of the Theta matrices with a(n) matrices
% can be done two ways, the one implemented below requiring
% fewer transformation

%a1 = [ones(m,1), X];
%a2 = [ones(m,1), transpose(sigmoid(Theta1*transpose(a1)))];
%a3 = transpose(sigmoid(Theta2*transpose(a2)));

a1 = [ones(m,1), X];
a2 = [ones(m,1), sigmoid(a1 * transpose(Theta1))];
a3 = sigmoid(a2 * transpose(Theta2));

delta2 = zeros([size(a2, 2)]); % just the column count (nodes - 1 per theta)
delta3 = zeros([size(a3, 2)]); % ditto

% compute the cost by looping over a3 values and accumulating the 
% costs for each output node (10 in this case), for each y 
for i = 1:m 
    % map y values to a row vector of num_labels length so that 
    % you can compare to output nodes while computing cost
    yvector = zeros(num_labels, 1);
    yvector(y(i), 1) = 1;
    for j = 1:num_labels
        % yvector should be a col vector of size num_labels
        % a3 is 2 dimensional but we're selecting the corresponding
        % sample row (i) and theta(j) one at a time.
        J += -yvector(j)*log(a3(i,j)) - (1 - yvector(j))*log(1 - a3(i,j));
    end

    % vectorized computation of delta3 (the easiest delta to compute)
    % (delta3 has one element per output node, so 10 elements)
    % you could also do this above inside the for loop
    % for each node one at a time (non-vectorized)
    % since a3 was computed for all samples, it's 5000x10
    % just take the current sample's row (the current m in the loop)
    delta3 = transpose(a3(i, :)) - yvector;  % element-wise subtraction
    % now d3 has deltas for just one sample
    % and we can back prop to d2 for that sample
    % transpose Theta2 is a 26x10; and, since you're only 
    % computing for a single m inside this loop,
    % you only want a single row from a2 which was computed
    % for all the samples before entry into this loop.
    % here's that single row to make this more readable:
    a2m = a2(i, :); % 1 x 26 since there are 10 output nodes in layer a2
    a1m = a1(i, :); % 1 X 401 - these are input nodes
    % transpose(Theta2) is a 26 x 10; delta3 is a 1 x 10 so
    % that multiplcation should produce a 26 x 10; a2m is a 1 x 26
    % how will that work with element-wise multiplcation?
    delta2 = (transpose(Theta2) * delta3) .* transpose(a2m .* (1 - a2m));
    % now lop off the bias column; there is no bias column
    % in delta3 because that was calculated from nn outputs a3
    delta2 = delta2(2:end);
    % now add these deltas to accumulaters that will be updated for 
    % every training row m and then used to compute partial 
    % derivates for all thetas for the network
    Theta2_grad = Theta2_grad + delta3 * a2m;
    Theta1_grad = Theta1_grad + delta2 * a1m;
end 

Theta2_grad = (1/m) .* Theta2_grad;
Theta1_grad = (1/m) .* Theta1_grad;


J = (1/m) * J;

% now add regularization (for all but the bias Thetas in the first columns
J += (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


% so now we have the cost of the entire network; it's a single value
% how does that help?  It helps because gradient descent
% and other optimized functions measure cost on each pass.


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
