function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% the distance matrix will be a m x k matrik.
% where each column reprsents the distance of the examples from the corresponding
% kth centroid.
distanceMatrix = zeros(size(X, 1), size(centroids, 1));

size(X); % 300 x 2
size(centroids); % 3 x 2
size(distanceMatrix); % 300 x 3

% for each centroid, get the distance to the example
for k=1:size(centroids, 1)
	columnVector = zeros(size(X, 1), 1);

	diffs = bsxfun(@minus, X, centroids(k, :));
	% do sums row wise
	columnVector = sum(diffs .* diffs, 2);
	distanceMatrix(:, k) = columnVector;
end

% calculate the index of the smallest value in each row of the distance matrix.
% distance matrix now contains the distance of each example to the centroids given
% x captures values and idx captures indices. indices is what we are interested in
[x, idx] = min(distanceMatrix, [], 2);

% =============================================================

end