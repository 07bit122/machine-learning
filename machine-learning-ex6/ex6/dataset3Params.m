function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

possibilities = [0.01 0.03 0.1 0.3 1 3 10 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% variables to save the final values
minError = inf;
cFinal = inf;
sigmaFinal = inf;

for i = possibilities
	for j = possibilities
		cVal = i;
		sigmaVal = j;

		trainedModel = svmTrain(X, y, cVal, @(x1, x2) gaussianKernel(x1, x2, sigmaVal));
		trainedPredictions = svmPredict(trainedModel, Xval);
		error_iteration = mean(double(trainedPredictions ~= yval));

		if (error_iteration < minError)
			minError = error_iteration;
			cFinal = cVal;
			sigmaFinal = sigmaVal;
		end
	end
end

% after computing the C and sigma values for which minimum error is found by the trained model
% set the return values accordingly.
C = cFinal;
sigma = sigmaFinal;
% =========================================================================

end
