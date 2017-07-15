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

size(X); %12 x 2

size(theta); %x2 x 1

hypothesisTerm = (X * theta) - y;
hypothesisFunction = sum(hypothesisTerm .* hypothesisTerm);

% for regularization, the first theta term should be ignored.
theta(1) = 0;
regularizationTerm = sum(theta .* theta);

J = (hypothesisFunction / (2 * m)) + ((regularizationTerm * lambda) / (2 * m));

for loop = 1:size(theta)(1)

	regularization = (theta(loop, 1) * lambda) / m;
	summation = (sum(hypothesisTerm .* X(:, loop))) / m;
	if (loop == 1)
		% DO NOTHING
    else
    	summation = summation + regularization;
    end
    grad(loop) = summation;
end

% =========================================================================

grad = grad(:);

end