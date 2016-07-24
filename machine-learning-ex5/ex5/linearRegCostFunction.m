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
tmp = 0;
The_1 = theta .^ 2;

for i = 2 : size(The_1,1)
	tmp = tmp + The_1(i);
end

regular = lambda * tmp / 2 / m;

J = sum((X*theta-y).^2)/ 2 / m + regular;


tmp = lambda * theta / m;

grad = X'*(X*theta - y) / m + tmp;

grad(1) = grad(1) - lambda * theta(1) / m;







% =========================================================================

grad = grad(:);

end
