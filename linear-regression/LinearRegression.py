# This is a program to perform multivariate linear regression on datasets with any amount of input variables. To use this program, format your data into two matrices. The first matrix should contain a row of ones followed by a row for each input variable in the dataset. The other matrix should be a 1D row vector with each column containing the output value for the corresponding input value. Ex. Input: [[1,1,1,...],[x1,x2,x3,...],...];  Output: [y1,y2,y3,...];

import random;
def initializeParameters(inVals):
    return [random.random()+1 for i in xrange(len(inVals))];

def prediction(inVals, params):
    predictions = [];
    for i in xrange(len(inVals[0])):
        predict = 0;
        for j in xrange(len(params)):
            predict += inVals[j][i]*params[j];
        predictions.append(predict);
    return predictions;

def costDerivative(inVals, outVals, params, row):
    predictions = prediction(inVals, params);
    sum_ = 0;
    for i in xrange(len(predictions)):
        sum_ += (predictions[i] - outVals[i])*inVals[row][i];
    return sum_;

# Potentially correct, not verified
def scale(inVals):
    for i in xrange(len(inVals)):
        mean = sum(inVals[i])/float(len(inVals[i]));
        minn = min(inVals[i]);
        rang = max(inVals[i]);
        if (rang != minn):
            rang = rang - minn;
        for j in xrange(len(inVals[i])):
            inVals[i][j] = (inVals[i][j]-mean)/float(rang);
    return inVals;

def cost(inVals, outVals, params):
    predictions = prediction(inVals, params);
    sum_ = 0;
    for i in xrange(len(predictions)):
        sum_ += (predictions[i] - outVals[i])**2;
    return sum_;

def optimizeParams(inVals, outVals, params, learningRate):
    prevParams = [100]*len(params);
    minimized = False;
    while(not minimized):
        print prediction(inVals, params);
        for i in xrange(len(params)):
            prevParams[i] = params[i];
            params[i] = params[i] - learningRate/len(inVals)*costDerivative(inVals, outVals, params, i);
        if(abs(cost(inVals, outVals, params)) < .000000001):
            minimized = True;
    return params;

## Tests show an accuracy of the parameterization of 99.99%.
inValues = [1,2,3,4,5,6,7,8,9];
outValues = [1,2,3,4,5,6,7,8,9];
expectedParams = [0,1];
learningRate = .001;
inValues = [[1]*len(inValues),inValues];
testParams = initializeParameters(inValues);
testParams = optimizeParams(inValues, outValues, testParams, learningRate);
print testParams;
