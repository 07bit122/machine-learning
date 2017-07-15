function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% computer the sigmoid function
sigmoidX = sigmoid( X * theta );
% this is the complement for sigmoid
sigmoidXcomplement = 1 .- sigmoidX;
% sigmoidX .+ sigmoidXComplement should be 1.

% in the cost function formula, this is the first term
firstTerm = log (sigmoidX) .* y;
% in the cost function formula, this is the second term
secondTerm = log (sigmoidXcomplement) .* (1 .- y);

J = sum(firstTerm .+ secondTerm) / -m;

% for calculating the gradients, we have to loop over each value of theta
% so that each value of grad could be updated and this also makes our lives
% easy in extracting the correct feature column in dataset X
for loop = 1:size(theta)(1)

    summation = sum((sigmoidX .- y) .* X(:, loop));
    grad(loop) = summation / m;
end

% =============================================================

end
