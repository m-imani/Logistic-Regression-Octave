function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


hx = X * theta;  
gx = sigmoid(hx);  

% Cost function J  
pos = -y' * log(gx);  
neg = (1 - y)' * log(1 - gx);  
regular = lambda / (2 * m) * sum(theta(2:end) .^ 2);  

J = (1 / m) * (pos - neg) + regular;  

% Gradient descent  

grad = (1/m) * X' * (gx - y) + (lambda/m) .* theta;  
grad(1) = (1/m) * X(:,1)' * (gx - y);  


end
