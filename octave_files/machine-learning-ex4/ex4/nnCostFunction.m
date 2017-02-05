function [cost costGradient] = nnCostFunction(neuralNetParams, ...
                                   inputSize, ...
                                   hiddenLayerSize, ...
                                   possiblePredictionsCount, ...
                                   X, y, lambda)
                                   
                                   
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [cost costGradient] = NNCOSTFUNCTON(neuralNetParams, inputSize, hiddenLayerSize, possiblePredictionsCount, ...
%   X, y, lambda) computes the cost and costGradientient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   neuralNetParams and need to be converted back into the weight matrices.
%
%   The returned parameter costGradient should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape neuralNetParams back into the parameters step1Weights and step2Weights, the weight matrices
% for our 2 layer neural network
step1Weights = reshape(neuralNetParams(1:hiddenLayerSize * (inputSize + 1)), ...
                 hiddenLayerSize, (inputSize + 1));
                

step2Weights = reshape(neuralNetParams((1 + (hiddenLayerSize * (inputSize + 1))):end), ...
                 possiblePredictionsCount, (hiddenLayerSize + 1));

sampleSize = size(X, 1);
formattedYs = zeros(possiblePredictionsCount,sampleSize);
for k = 1:sampleSize
  formattedYs(y(k),k) = 1;
end

% Initialize goal variables
cost = 0;
step1WeightsCostGradient = zeros(size(step1Weights));
step2WeightsCostGradient = zeros(size(step2Weights));


%% Feed forward algorithm:
% Calculates neuralNet prediction
X = [ones(16,1),X];
hiddenLayerPrediction = sigmoid(step1Weights*X');
hiddenLayerPredictionWithBias =  [ones(1,size(hiddenLayerPrediction,2));hiddenLayerPrediction];
neuralNetPredictionMatrix = sigmoid(step2Weights*hiddenLayerPredictionWithBias);

% Calculates cost based off of prediction
cost = -(1.0/sampleSize)*(formattedYs.*log(neuralNetPredictionMatrix) + (1 - formattedYs).*log(1 - neuralNetPredictionMatrix));
cost = sum(cost,1);
cost = sum(cost);

% Part 2: Implement the backpropagation algorithm to compute the costGradientients
%         step1Weights_costGradient and step2Weights_costGradient. You should return the partial derivatives of
%         the cost function with respect to step1Weights and step2Weights in step1Weights_costGradient and
%         step2Weights_costGradient, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNcostGradientients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

%% Backpropagation
% Calculates error in prediction outputs
predictionErrorMatrix = formattedYs - neuralNetPredictionMatrix;
hiddenLayerErrorMatrix = (step2Weights'*predictionErrorMatrix) .* hiddenLayerPredictionWithBias .* (1-hiddenLayerPredictionWithBias);

for l = 1:sampleSize
  for i = 1:size(step1Weights,1)
    for j = 1:size(step1Weights,2)
      step1WeightsCostGradient(i,j) = step1WeightsCostGradient(i,j) + hiddenLayerErrorMatrix(i,l)*X'(j,l); 
    end
  end
end

step1WeightsCostGradient(:,1) = (1.0/sampleSize)*step1WeightsCostGradient(:,1);
step1WeightsCostGradient(:,2:end) = (1.0/sampleSize)*(step1WeightsCostGradient(:,2:end)+lambda*step1Weights(:,2:end));

for l = 1:sampleSize
  for i = 1:size(step2Weights,1)
    for j = 1:size(step2Weights,2)
      step2WeightsCostGradient(i,j) = step2WeightsCostGradient(i,j) + predictionErrorMatrix(i,l)*hiddenLayerPredictionWithBias(j,l); 
    end
  end
end
step2WeightsCostGradient(:,1) = (1.0/sampleSize)*step2WeightsCostGradient(:,1);
step2WeightsCostGradient(:,2:end) = (1.0/sampleSize)*(step2WeightsCostGradient(:,2:end)+lambda*step2Weights(:,2:end));


% Includes regularization in cost
regularizationCost = lambda/(2*sampleSize)*(sum(sum(step1Weights(:,2:end).*step1Weights(:,2:end))) + ...
                        sum(sum(step2Weights(:,2:end).*step2Weights(:,2:end)))); % Must ignore bias terms
cost = cost + regularizationCost;



% Part 3: Implement regularization with the cost function and costGradientients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the costGradientients for
%               the regularization separately and then add them to step1Weights_costGradient
%               and step2Weights_costGradient from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll costGradientients
costGradient = [step1WeightsCostGradient(:) ; step2WeightsCostGradient(:)];


end
