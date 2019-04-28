function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
% I don't understand the value of jamming the X and Theta arrays into params 
% only to have to pull them back out immediately inside the function
% Why not just pass the as separate params?  Yes, they're both have
% num_features cols.  Is that why?  If so, still don't see it.
% Oh, it's so it'll work with fminunc, or other optimized functions
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% you can find cost of both X and Theta at the same time (i.e. collaborative filtering)
% remember this is just the cost function for passed values of Theta and X, it's not the
% partial derivative of the cost.  We'll calculate that next.

% Theta is users parameters, NOT specific to a movie, just how they value particular 
% features as they get defined.
% X is features for all movies
% Y is where the ratings are, movies x users


% feels like this should be vectorized
% need to sum only where the user has rated the movie
% see below for vectorized version!!
%for i = 1:num_movies
%    for j = 1:num_users
%        if R(i,j) == 1
%            % this isn't theta transpose x but x theta transpose in order to get
%            % the vectors looking like: 1x3 * 3x1 so that the result is 1x1
%            J = J + 1/2 * (X(i, :) * transpose(Theta(j, :))- Y(i,j))^2
%        end
%    end
%end

% X * transpose(Theta) will yield a num_movies x num_users matrix;
% Y is also num_movies x num_users - so subtracting that is very simple
% likewise, R is also num_movies x num_users so to negate users+movies without ratings, just use dot product
% then square every element and sum in both directions to produce the cost
J = 1/2 * sum(sum(((X * transpose(Theta) - Y) .* R).^2)) + ((lambda/2) * sum(sum(Theta.^2))) + ((lambda/2) * sum(sum(X.^2)));


% same basic X * transpose(Theta) as above but since these are partial derivatives, no squaring,
% no 1/2 and need to multiply by one or the other term 
% this is a vectorized version
% also remember that this is just a gradient, not the formula to calculate the next Theta or X value\
% like usual, focus on aligning the rows columns in the matrix multiplications to what you need 
% in calculating these two grads, you want the same dimensions as the starting X and Theta matrices:
% movies x features  and   users by features, respectively
X_grad = ((X * transpose(Theta) - Y) .* R) * Theta .+ (lambda .* X);
Theta_grad = transpose(((X * transpose(Theta) - Y) .* R)) * X .+ (lambda .* Theta);

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
