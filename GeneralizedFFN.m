%Stochastic Gradient Descent with Backpropagation and Biases
%script variables
inputNum = 8;
outputNum = 10;
hiddenNodes = 3;
hiddenLayers = 3;
eta = .5; %learning rate
epochs = 200;
my_data = data;
my_output = classes;

%sigmoid function
g = @(x, w) 1/(1 + exp(-w'*x));

randArr = randperm(1484);
training_indices = randArr(1:round(1484*.65));
testing_indices = randArr(round(1484*.65) + 1 : 1484);
X = horzcat(ones(1484, 1), my_data);

training_X = X(training_indices(:), :);
training_Y = my_output(training_indices(:), :);
[training_size, ~] = size(training_X);
testing_X = X(testing_indices(:), :);
testing_Y = my_output(testing_indices(:), :);
        
%Creating the weight matrices (must add 1 for bias)
cur_W_layer{1} = rand(hiddenNodes, inputNum+1);
next_W_layer{1} = rand(hiddenNodes, inputNum+1);

for l = 2:hiddenLayers+1
	if l == hiddenLayers + 1
		cur_W_layer{l} = rand(outputNum, hiddenNodes+1);
		next_W_layer{l} = rand(outputNum, hiddenNodes+1);
	else
		cur_W_layer{l} = rand(hiddenNodes, hiddenNodes+1);
		next_W_layer{l} = rand(hiddenNodes, hiddenNodes+1);
	end
end

%Creating the activation output of each layer (column vector)
for l = 1:hiddenLayers+1
	if l == hiddenLayers+1
		a_layer{l} = ones(outputNum,1);
	else
		a_layer{l} = ones(hiddenNodes+1,1);
	end
end

%Create delta matrix used for updating (column vectors)
for l = 1:hiddenLayers+1
	if l == hiddenLayers+1
		delta{l} = zeros(outputNum, 1);
	else
		delta{l} = zeros(hiddenNodes, 1);
	end
end

%starting FFN iterations
for n = 1:epochs
	for m = 1:training_size

		%calculates activation values for first hidden layer
		for i = 2:hiddenNodes+1
			a_layer{1}(i,1) = g(cur_W_layer{1}(i-1, :)', training_X(m, :)');
		end
		
		%calculates activation values for other hidden layers
		%does not include output layer
		if hiddenLayers > 1
			for l = 2:hiddenLayers
				for j = 2:hiddenNodes+1
					a_layer{l}(j,1) = g(cur_W_layer{l}(j-1, :)', a_layer{l-1}(:));
				end
			end
		end
		
		%calculate activation values for the output
		for k = 1:outputNum
			a_layer{hiddenLayers+1}(k,1) = g(cur_W_layer{hiddenLayers+1}(k,:)', a_layer{hiddenLayers}(:));
		end
		
		%delta function for output layer
		dK = @(yk, ak) (yk - ak)*(1 - ak)*(ak);
		
		%calculate derivative of RSS with respect to last hidden layer weights
		%and update last hidden layer weights for next iteration
		for k = 1:outputNum
			delta{hiddenLayers + 1}(k) = dK(training_Y(m,k), a_layer{hiddenLayers + 1}(k,1));
			
			%+1 from previous layer nodes and the bias node
			for j = 1:hiddenNodes + 1
				next_W_layer{hiddenLayers + 1}(k,j) = cur_W_layer{hiddenLayers + 1}(k,j) + eta * delta{hiddenLayers+1}(k) * a_layer{hiddenLayers}(j,1);
			end
		end
		
		%delta function for layers below output
		dK2 = @(deltaSum, aj) deltaSum*(1 - aj)*(aj);
		
		for l = hiddenLayers:-1:2
			for j = 1:hiddenNodes
				sumOfDelta = delta{l+1}(:)' * cur_W_layer{l+1}(:, j+1);
				delta{l}(j) = dK2(sumOfDelta, a_layer{l}(j+1,1));
				for i = 1:hiddenNodes+1
					next_W_layer{l}(j,i) = cur_W_layer{l}(j,i) + eta * delta{l}(j) * a_layer{l-1}(i);
				end
			end
		end
		
		%weights from the input to the first hidden layer
		for j = 1:hiddenNodes
			sumOfDelta = delta{2}(:)' * cur_W_layer{2}(:, j+1);
			delta{1}(j) = dK2(sumOfDelta, a_layer{1}(j+1,1));
			for i = 1:inputNum+1
				next_W_layer{1}(j,i) = cur_W_layer{1}(j,i) + eta * delta{1}(j) * training_X(m,i);
			end
		end
		
		%update the weight layers
		for l = 1:hiddenLayers+1
			cur_W_layer{l}(:,:) = next_W_layer{l}(:,:);
		end
	end
end