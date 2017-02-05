function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
indySum = @(x,y,theta) (x*theta - y);


for iter = 1:num_iters
    dTheta0 = 0;
    dTheta1 = 0;
    for k = 1:m
        dTheta0 = dTheta0 + indySum(X(k,:),y(k),theta);
        dTheta1 = dTheta1 + X(k,2)*indySum(X(k,:),y(k),theta);
    end
    theta(1) = theta(1) - alpha/m * dTheta0;
    theta(2) = theta(2) - alpha/m * dTheta1;
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
