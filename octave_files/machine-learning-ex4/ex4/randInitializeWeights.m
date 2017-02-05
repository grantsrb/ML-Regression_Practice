function randomWeights = randInitializeWeights(columns, rows)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with columns
%incoming connections and rows outgoing connections
%   randomWeights = RANDINITIALIZEWEIGHTS(columns, rows) randomly initializes the weights 
%   of a layer with columns incoming connections and rows outgoing 
%   connections. 
%
%   Note that randomWeights is a matrix of size(L_out, 1 + L_in) as
%   the first column of randomWeights handles the "bias" terms
%


randomWeights = rand(rows, 1 + columns) * (2 * 0.12) - (0.12);


end
