function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
for loop = 1:size(theta)(1)

	regularization = (theta(loop, 1) * lambda) / m;
	summation = (sum((sigmoidX .- y) .* X(:, loop))) / m;
	if (loop == 1)
		% DO NOTHING
    else
    	summation = summation + regularization;
    end
    grad(loop) = summation;
end

% =============================================================

end
