function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for the 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1 = [ones(m,1),X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2*Theta2';
htheta = sigmoid(z3);

for k = 1:num_labels
    yk = y == k;
    hthetak = htheta(:,k);
    temp = 1/m*sum((-yk.*log(hthetak))-(1-yk).*log(1-hthetak));
    J = J + temp;
end

reg = lambda/(2*m)*(sum(sum(Theta1(:, 2:end) .^ 2))+sum(sum(Theta2(:, 2:end) .^ 2)));
J = J + reg;
    

for i = 1:m
    for k = 1:num_labels
        yk = y(i) == k;
        delta3(k) = htheta(i,k)-yk;
    end
    delta2 = Theta2' * delta3' .* sigmoidGradient([1, z2(i, :)])';
    delta2 = delta2(2:end);
    
    Theta1_grad = Theta1_grad + delta2 * a1(i, :);
    Theta2_grad = Theta2_grad + delta3' * a2(i, :);
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
