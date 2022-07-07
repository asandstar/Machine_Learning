function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------
I=eye(num_labels);
Y=zeros(m,num_labels);
for i=1:m
    Y(i,:)=I(y(i),:);
end
a1=[ones(m, 1) X];
z2=a1*Theta1';
a2=[ones(size(z2, 1),1) sigmoid(z2)];
z3=a2*Theta2';
a3=sigmoid(z3);

h = a3;
r = sum(sum(Theta1(:,2:end).^2,2))+sum(sum(Theta2(:,2:end).^2,2));
J = sum(sum((-Y).*log(h)-(1-Y).*log(1-h),2))/m+lambda*r/(2*m);
sigma3=a3-Y;
sigma2=(sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1),1) z2]);
sigma2=sigma2(:,2:end);
delta_1=(sigma2'*a1);
delta_2=(sigma3'*a2);

p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad=delta_1./m+p1;
Theta2_grad=delta_2./m+p2;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

%给你一个（全是废话的）版本如下

% % Part 1: Feedforward the neural network and return the cost in the
% %         variable J. After implementing Part 1, you can verify that your
% %         cost function computation is correct by verifying the cost
% %         computed in ex4.m
% %
% 
% I=eye(num_labels);
% %num_labels=10
% Y=zeros(m,num_labels);
% %全部为0的矩阵
% for i=1:m
%     Y(i,:)=I(y(i),:);
% end
% %放进command window可以看到上面输出一行九（？）列的矩阵
% 
% a1=[ones(m, 1) X];%重塑矩阵样式
% z2=a1*Theta1';%神经网络训练，赋权重之类的
% a2=[ones(size(z2, 1),1) sigmoid(z2)];%重塑矩阵样式，全部归一化进入0-1之内
% z3=a2*Theta2';%神经网络训练，赋权重之类的
% a3=sigmoid(z3);%全部归一化进入0-1之内
% 
% h = a3;
% %我就是代价函数的样本
% r = sum(sum(Theta1(:,2:end).^2,2))+sum(sum(Theta2(:,2:end).^2,2));
% %上面是那个公式的理解，后面两个求和分别对应theta1和theta2
% J = sum(sum((-Y).*log(h)-(1-Y).*log(1-h),2))/m+lambda*r/(2*m);
% %计算代价函数的公式
% 
% % Part 2: Implement the backpropagation algorithm to compute the gradients
% %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
% %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% %         Theta2_grad, respectively. After implementing Part 2, you can check
% %         that your implementation is correct by running checkNNGradients
% %
% %         Note: The vector y passed into the function is a vector of labels
% %               containing values from 1..K. You need to map this vector into a 
% %               binary vector of 1's and 0's to be used with the neural network
% %               cost function.
% %
% %         Hint: We recommend implementing backpropagation using a for-loop
% %               over the training examples if you are implementing it for the 
% %               first time.
% %
% 
% sigma3=a3-Y;
% %好激动，反向传播算法开始了！！！
% sigma2=(sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1),1) z2]);
% %诶这就是计算公式罢了,(θn转置*σn+1)*g'(z3),只不过z3需要重排一下形式
% sigma2=sigma2(:,2:end);
% %好嘞继续重排sigma2
% delta_1=(sigma2'*a1);
% delta_2=(sigma3'*a2);
% % Part 3: Implement regularization with the cost function and gradients.
% %
% %         Hint: You can implement this around the code for
% %               backpropagation. That is, you can compute the gradients for
% %               the regularization separately and then add them to Theta1_grad
% %               and Theta2_grad from Part 2.
% %
% %下面就是咱的正则化了！
% p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
% p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
% Theta1_grad=delta_1./m+p1;
% Theta2_grad=delta_2./m+p2;

