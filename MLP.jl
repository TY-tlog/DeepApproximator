###################################################################################
############################# Data Loading ########################################
###################################################################################
using JLD2
@load "/Volumes/YD/DeepApproximator/lorenz_data.jld2" t Y

input=Y[1,1:60]
target=Y[2,1:60]

###################################################################################
########################## single layer, single node MLP ##########################
###################################################################################

# define activation function(sigmoid)
f(x) = 1 / (1 + exp(-x))
f_derivative(x) = f(x) * (1 - f(x))

# Initialize weights and biases
#input to hidden
w1 = rand(1, length(input)) 
b1 = rand(1, 1)
#hidden to output
w2 = rand(length(target), length(w1[:,1])) 
b2 = rand(1, 1)

# forward pass
function forward(x)
    h = f.(w1 * x .+ b1)
    y_pred = w2 * h .+ b2
    return y_pred, h
end

# backpropagation
function backprop(x, y, w1, b1, w2, b2, learning_rate)
    y_pred, h = forward(x)              # 예측값과 은닉층 출력 계산
    loss = sum(y_pred .- y)/length(y)
    error = y_pred .- y
    d_y_pred = error                    # 출력층에서의 손실에 대한 미분값

    # 출력층 가중치와 바이어스에 대한 그라디언트 계산
    d_w2 = d_y_pred * h'
    d_b2 = d_y_pred

    # 은닉층 가중치와 바이어스에 대한 그라디언트 계산
    d_w1 = w2' * d_y_pred .* f_derivative.(w1 * x .+ b1) .* x'
    

    # 가중치와 바이어스 업데이트
    w1 -= learning_rate * d_w1
    w2 -= learning_rate * d_w2    
    return w1, w2, loss
end

backprop(input, target, w1, b1, w2, b2, 0.01)

n_epochs = 1
learning_rate = 0.01

n_epochs = 1000

for epoch in 1:n_epochs
    w1, w2, loss = backprop(input, target, w1, b1, w2, b2, learning_rate)
end
loss

h = f.(w1 * input .+ b1)
y_pred = w2 * h .+ b2