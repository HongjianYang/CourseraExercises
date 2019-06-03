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

% First step: Compute the unregulatized cost function
% Compute the hypothesis
h = X * theta;
J = sum((h - y) .^ 2) / ( 2 * m );

% Next, add the regularization term
partial_theta = theta(2: size(theta, 1), : );
reg_term = sum(partial_theta .^ 2) * lambda / ( 2 * m );
J = J + reg_term;

% Then, compute the unregularized gradient
% Dimension of X: 12 * 2, dimension: (h - y): 12 * 1
grad = X' * (h - y) / m;

% Finally, compute the regularized gradient
temp = grad(2: size(grad, 1), :);
temp = temp + (lambda / m) * partial_theta;
grad(2: size(grad, 1), :) = temp;









% =========================================================================

grad = grad(:);

end
