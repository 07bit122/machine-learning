function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % initialize temp to be a matrix of same dimensions as theta
    temp = theta;
    % compute the next values of each element in the theta matrix
    % by following the gradient descent algorithm.
    % save them in temporary variables.
    for loop = 1:size(theta)(1)
        s = sum(((X * theta) - y) .* X(:, loop));
        a = alpha * s;
        temp(loop) = temp(loop) - (a/m);
    end

    % ============================================================

    % Save the cost J in every iteration
    % compute the cost function based on the present
    % values of theta
    J_history(iter) = computeCost(X, y, theta);

    % if at this point theta and temp are still the same
    % it means that the values of theta have converged
    % and we attained the minimum in our gradient descent
    if (theta == temp)
        break;
    else
        % otherwise, update theta to reflect temp
        theta = temp;
    endif

end

end
