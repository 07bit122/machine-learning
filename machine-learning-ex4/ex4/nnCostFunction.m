function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_matrix = eye(num_labels)(y, :);

% corresponds to the first layer
aone = [ones(m, 1) X];

size(Theta1); % 25 x 401
size(aone); % 5000 x 401

% corresponds to the second layer
ztwo = aone * Theta1';

atwo = sigmoid(ztwo);

atwo = [ones(size(atwo, 1), 1) atwo];

size(atwo); % 5000 x 26

% corresponds to the third layer
zthree = atwo * Theta2';

% athree is h(theta)
athree = sigmoid(zthree);

% in logistic regression, a complement is basically 1 - x
athreecomplement = 1 .- athree;

size(athree); % 5000 x 10

% in the cost function formula, this is the first term
firstTerm = log(athree) .* y_matrix;

% in the cost function formula, this is the second term
secondTerm = log(athreecomplement) .* (1 .- y_matrix);

summationMatrix = zeros(m, 1);

% sum over the k terms and change these from 5000 x 10 matrices to 5000 * 1 matrices.
for loop=1:m
	firstTermSubSet = firstTerm(loop, :);
	secondTermSubSet = secondTerm(loop, :);
	summationMatrix(loop, 1) = sum(firstTermSubSet .+ secondTermSubSet);
end

% after the for loop, summationMatrix would be 5000 x 1 matrix.

% do a summation of all the 5000 values and divide it by -m.
% this is unregularized cost function
J = sum(summationMatrix) / -m;

% add regularization parameters

% thetaone regularization
% remove the bias column from theta1.
rowsinTheta1 = size(Theta1)(1);
columnsinTheta1 = size(Theta1)(2);

% this is a vector of Theta1 excluding all the values in column1 of Theta1
Theta1DeepCopy = Theta1(rowsinTheta1+1:end);

% same as Theta1 excluding the first column.
Theta1DeepCopy = reshape(Theta1DeepCopy, rowsinTheta1, columnsinTheta1-1);
Theta1DeepCopy = Theta1DeepCopy .* Theta1DeepCopy;
Theta1Sum = sum(sum(Theta1DeepCopy));

% thetatwo regularization
% remove the bias column from theta1.
rowsinTheta2 = size(Theta2)(1);
columnsinTheta2 = size(Theta2)(2);

% this is a vector of Theta1 excluding all the values in column1 of Theta1
Theta2DeepCopy = Theta2(rowsinTheta2+1:end);

% same as Theta1 excluding the first column.
Theta2DeepCopy = reshape(Theta2DeepCopy, rowsinTheta2, columnsinTheta2-1);
Theta2DeepCopy = Theta2DeepCopy .* Theta2DeepCopy;
Theta2Sum = sum(sum(Theta2DeepCopy));

J_Regular = ((Theta1Sum + Theta2Sum) * lambda) / (2 * m);

J = J + J_Regular;
% -------------------------------------------------------------

% start back propagation
delta3 = athree - y_matrix; % 5000 x 10

delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(ztwo); % 5000 x 25

size(delta2); % 5000 x 25

Delta1 = delta2' * aone;
Delta2 = delta3' * atwo;

% add regularization
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
scaledTheta1 = Theta1 .* (lambda / m);
scaledTheta2 = Theta2 .* (lambda / m);

Theta1_grad = (Delta1 / m) .+ scaledTheta1;
Theta2_grad = (Delta2 / m) .+ scaledTheta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
