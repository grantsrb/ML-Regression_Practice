function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. The extra regularization
%   term does not act on the first element in theta in both the cost and
%   the gradient.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J = 1/m*(sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)))+lambda/2*sum(theta(2:end).^2));

for k = 1:size(theta)
    if(k == 1)
        grad(k) = 1/m*sum((sigmoid(X*theta)-y).*X(:,k));
    else
        grad(k) = 1/m*sum((sigmoid(X*theta)-y).*X(:,k))+lambda/m*theta(k);
    end
end
% =============================================================

end
