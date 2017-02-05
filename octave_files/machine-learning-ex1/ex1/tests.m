% %% Optional Exercise
% clc;
% clf;
% clear;
% numIters = 50;
% theta = zeros(3, 1);
% data = load('ex1data2.txt');
% X = data(:, 1:2);
% y = data(:, 3);
% m = length(y);
% alpha = 1;
% 
% [X mu sigma] = featureNormalize(X);
% X = [ones(m, 1) X];
% 
% for k = 1:6
%     [theta jHistory] = gradientDescentMulti(X,y,theta,alpha,numIters);
%     figure
%     plot(1:50, jHistory);
%     title(num2str(alpha));
%     
%     if mod(k,2) == 1
%         alpha = alpha*3;
%     else
%         alpha = (alpha/3)*10^-1;
%         disp(alpha);
%     end
% end

x = [1 1 1; 1 1 1];
log(x)
