function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
len = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid( X * theta);
one = ones(m, 1);
J = sum( -y .* log( h ) - ( one - y ) .* log( one - h ) ) / m + (lambda / (2 * m)) * sum(theta .^ 2);

sample0 = X(:, 1);
sample = X(:, 2:len);

grad(1,1) = sample0' * ( h - y ) / m;
grad(2:len ,1) = sample' * ( h - y ) / m + (lambda / m) * theta(2:len, 1);
 



% =============================================================

end
