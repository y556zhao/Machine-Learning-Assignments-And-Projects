function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for x=1:size(z,1)
  for y=1:size(z,2)
    g(x,y)=1/(1+(e.^(-z(x,y))));
endfor
endfor

% =============================================================

end
