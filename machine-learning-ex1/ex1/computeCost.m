function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Compute the cost function
% .* does a number to number multiplication
% sum adds up all the individual elements of the matrix
% add a semicolon at the end so that the cli doesn't get cluttered
J = sum(((X * theta) - y) .* ((X * theta) - y)) / (2 * m);


% =========================================================================

end
