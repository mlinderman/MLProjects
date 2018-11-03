function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

fprintf('\nTheta1: ');
size(Theta1); % a2 inputs (25 * n+1)  - just remove semicolon to debug 

fprintf('\nTheta2: ');
size(Theta2); % a3 inputs (10 x n+1)

fprintf('\nX: ');
size(X); % m x n

X = [ones(m, 1) X];
fprintf('\nX: ');
size(X);

% had to ponder this for quite some time and then    I
% had to write it down to visualize what I wanted:
% basically, you're creating a new X with 
% 25 rows and 401 columns (from 5000 rows and 401 columns)

% Theta1:
% 43, 54, 65, 67, 86 ...........  401
% 23, 54, 56, 67, 39 ...........  401
% .
% .
% .
% 25

% X:
% 1234, 34, 64, 56, 57 .........  401
% 4332, 12, 54, 22, 33 .........  401
% .
% .
% .
% 5000


% seems the desired result will reduce
% the inputs from 401 features down to 25 (the number of inputs).
% So, the number of rows in Theta1 needs to 
% become the number of Thetas in a2, a column vector
a2 = sigmoid(Theta1 * transpose(X)); % sum the rows

% add 1's 
a2 = [ones(1, size(a2, 2)); a2];

% now compute a3
a3 = transpose(sigmoid(Theta2 * a2));

% now round and max to find column index of prediction (which should align with the digit predicted)
[val, p] = max(a3, [], 2);

% p contains some 10s which should be zeros since column 10 was mapped to zero but looks like
% the caller is taking that into account.

% =========================================================================


end
