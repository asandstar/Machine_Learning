function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

results=[];
for pC=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for psigma=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        model = svmTrain(X, y, pC, @(x1, x2)gaussianKernel(x1, x2, psigma));
        %原来是这里出现了错误不应该是C和sigma而应该是测试集
        predictions = svmPredict(model, Xval);
        error=mean(double(predictions ~= yval));
        %fprintf("C: %f\nsigma: %f\nerror: %f\n", pC, pSigma, error);
        results=[results;pC,psigma,error];
    end
end
[merror,i]=min(results(:,3));
C=results(i,1);
sigma=results(i,2);
% [C,I] = min(...)找到A中那些最小值的索引位置，将他们放在向量I中返回。
% 如果这里有多个相同最小值时，返回的将是第一个的索引。

fprintf("\n\nLeast error:\nC: %f\nsigma: %f\nerror: %f\n", C, sigma, merror);
% =========================================================================

end
