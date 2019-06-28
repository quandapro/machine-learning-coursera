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
h = X*theta;

J = sum((h - y).^2)/2/m + lambda*sum(theta(2:size(theta, 1)).^2)/2/m;
for j=1:size(grad, 1)
  for i=1:m
    grad(j) += (h(i) - y(i))*X(i, j)/m;
  endfor
  if (j != 1) 
    grad(j) += lambda*theta(j)/m;  
  endif
endfor



% =========================================================================

grad = grad(:);

end
