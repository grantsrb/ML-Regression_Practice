function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
derivative = zeros(1,size(X,2));

cost = @(x,y,theta) (x*theta - y);

for iter = 1:num_iters
    
    for k = 1:size(X,2)
        for j = 1:m
            derivative(k) = derivative(k) + X(j,k)*cost(X(j,:),y(j),theta);
        end
        derivative(k) = derivative(k)/m;
    end
    
    for k = 1:size(X,2)
        theta(k) = theta(k) - alpha * derivative(k);
    end

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
