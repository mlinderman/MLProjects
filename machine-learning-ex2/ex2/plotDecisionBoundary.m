function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])

%    marksDecisionBoundaryPlot(theta, X, y)
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end



function marksDecisionBoundaryPlot(theta, X, y)
% mark's attempt at plotting the real decision boundary
% since he doesn't understand the above and a line
% isn't very interesting (or accurate) - (result may overfit
% if inputs weren't regularized)
% for all possible integer x values within the range of
% feature 1 values (test score 1, column 2 in X), and all
% possible integer values of y within the range of 
% feature 2 (test score 2, column 3 in X), create a point
% each intersection of those where X*theta = 0
range_min = min([min(X(:,2)), min(X(:,3))]); 
range_max = max([max(X(:,2)), max(X(:,3))]);
% fake_X = [ones(range_max - range_min + 1, 1), transpose(range_min:range_max), transpose(range_min:range_max)]
% shitty idea, this was.  This will get you points on a single line from lower left to upper right, not all
% the points in the grid; what you really need is a fake_X that has every combination of x and y axis values
% over both ranges of x and y.  You could build that but better to just iterate and create points on the fly
% and plot those that have 0X = 0 as decision boundary points
for i = range_min:range_max
    for j = range_min:range_max
        xtheta = [1, i, j]*theta;
        if (xtheta < 0.5 && xtheta > -0.5)
            plot(i, j, "b")
        endif
    end
end
% well, that works but a) it's slow and b) it basically maps the same line as the code above, but 
% wider since I'm looking for a range of values close to zero.  Why is it a straight line?  The code above assumes
% a line, I think, so only takes 2 points.  But I thought my approach would get something a bit more aligned with the
% data and curved. But since none of the features has exponents, maybe a straight line is expected?  You could
% probably find that out if you changed one of the features...


end



hold off

end
