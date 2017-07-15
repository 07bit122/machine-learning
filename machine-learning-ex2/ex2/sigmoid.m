function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Calculate the exponentiation of each element of z.
% z could be a scalar, a matrix or a vector
exponent = exp(-z);

% do a element wise division on each element of z.
g = 1 ./ (1 + exponent);

% =============================================================

end
