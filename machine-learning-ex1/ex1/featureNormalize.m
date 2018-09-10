function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it''s standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X);       % mean() function will calculate the mean for each column in a matrix (nice!)
                   % but, if you have only a single row - like when you''re estimating for a single
                   % row of data (after you have thetas from the training set) it will return one value
                   % - same for std() - what to do?)
sigma = std(X);      % same for std()!

% feels like there should be some way to avoid this loop...
% maybe if you try calculating by row, across features, rather than by column (feature)
% didn''t do that here, obviously
% but SEE BELOW!
% for f=1:size(X, 2)  % for each feature of X (column)
    % for each value in X_norm column f, subtract then divide by sigma
%    X_norm(:, f) = (X_norm(:, f) - mu(f))./sigma(f);
%end

% why not this in place of the loop above?
% this will subtract a row vector from EACH row of X_norm, aligning feature values
% with their means. Then, dividing by each of those values by corresponding sigma values
% (using './' and not just '/' because multiplication and division do matrix mult/div, not 
% element wise, which is what you want here.  What''s weird is that row counts don''t match
% but if that''s the case, looks like as long as columns do, either of these operations will be 
% performed against each row)
X_norm = (X_norm - mu) ./ sigma;


% ============================================================

end
