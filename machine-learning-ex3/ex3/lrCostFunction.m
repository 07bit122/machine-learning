function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
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

% compute the regularization term
% we should always ignore the 1st term of theta during regularization
theta(1) = 0;
regularizationTerm = ((sum(theta .* theta)) * lambda) / (2 * m);

% the cost function should add the regularization term as well
J = (sum(firstTerm .+ secondTerm) / -m) + regularizationTerm;

% for calculating the gradients, we have to loop over each value of theta
% so that each value of grad could be updated and this also makes our lives
% easy in extracting the correct feature column in dataset X
sigmoidXminusY = sigmoidX .- y;

grad = grad .+ ((X' * sigmoidXminusY) / m) .+ ((theta * lambda) / m);

% =============================================================

grad = grad(:);

end