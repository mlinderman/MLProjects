function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
% so, euclidian difference between x1 and x2 the quantity squared all over sigma squared
% but... x1 and x2 are three element vectors in the data passed from ex6.m....
% those must be single samples, not features.  The fact that they're named x1 and x2
% is a bit confusing, however.  Yes, they are samples so you could have 
% n values in those vectors - but they should have the same lengths because
% they're both just samples.  

diffs = (x1 - x2).^2;
sim = exp(-sum(diffs)/(2*sigma^2));







% =============================================================
    
end
