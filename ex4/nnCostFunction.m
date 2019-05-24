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


% PART1: compute the cost
% Add the bias term to X, X now has the dimension 5000 * 401
X = [ones(size(X,1), 1) X];
% Get Z1, z1 has dimension 5000 * 25
z1 = X * Theta1';
% Get the activation from the hidden layer
a1 = sigmoid(z1);
% Add the biase term to a1, a1 has dimension 5000 * 26
a1 = [ones(size(a1, 1), 1) a1];
% Get z2, z2 has dimension 5000 * 10
z2 = a1 * Theta2';
% Get the output, h has dimension 5000 * 10
h = sigmoid(z2);

% Map vector y to a matrix of 5000 * 10 to compute cost function
yhat = zeros(m, num_labels);
for i = 1:m
    yhat(i, y(i)) = 1;
end

% Compute the cost of the neural network
% Firstly, compute cost for one single input
one = ones(m, num_labels);
difference = -yhat .* log(h) - ( one - yhat ) .* log( one - h );
% Then, sum these costs together
sum_one = sum(difference');
J = sum(sum_one');
J = J / m;


% PART2: Regulatized Cost Function
reg_theta1 = Theta1(:, 2: size(Theta1, 2));
reg_theta2 = Theta2(:, 2: size(Theta2, 2));

% Compute the sum of theta1 squared
sum_theta1 = sum(sum(reg_theta1 .^ 2)');
sum_theta2 = sum(sum(reg_theta2 .^ 2)');

% Compute the regulatization term
regularized = lambda / (2 * m) * (sum_theta1 + sum_theta2);

% Sum up the unregularized cost function and regularization term
J = J + regularized;

% Part 3: Compute the unregularized gradient
for i = 1:m
    partial_h = h(i,:)';
    partial_y = yhat(i,:)';
    partial_z = z1(i,:)';
    % Compute the delta in the output layer, dimension of diff3: 10 * 1
    diff3 = partial_h - partial_y;
    
    % Compute the delta in the second layer, dimension of diff: 25 * 1
    intermediate = Theta2' * diff3;

    diff2 = intermediate(2: size(intermediate, 1), 1) .* sigmoidGradient(partial_z);
    
    % Accrue the error, dimension of Theta1_grad: 25 * 401
    Theta1_grad = Theta1_grad + diff2 * X(i, :);
    % Dimension of Theta2_grad: 10 * 26
    Theta2_grad = Theta2_grad + diff3 * a1(i, :);
end
% This is the error term for the second layer
Theta1_grad = Theta1_grad / m;
% This is the error term for the third layer
Theta2_grad = Theta2_grad / m;

% Part 4: Compute the regularized gradient descent
% The first column can't be penalized as they are the bias neurons
temp1 = Theta1_grad(:, 2:input_layer_size + 1);
temp2 = Theta2_grad(:, 2:hidden_layer_size + 1);

% Add the regulatization term to the gradient
temp1 = temp1 + (lambda / m) * reg_theta1;
temp2 = temp2 + (lambda / m) * reg_theta2;

% Update the gradients

Theta1_grad(:, 2:input_layer_size + 1) = temp1;
Theta2_grad(:, 2:hidden_layer_size + 1) = temp2;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
