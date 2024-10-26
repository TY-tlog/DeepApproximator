###################################################################################
########################## single layer, single node MLP ##########################
###################################################################################

# define activation function(sigmoid)
σ(x) = 1 / (1 + exp(-x))

# Initialize weights and biases
#input to hidden
w1 = rand(1, 1) 
b1 = rand(1)
#hidden to output
w2 = rand(1, 1) 
b2 = rand(1)

# forward pass
function forward(x)
    h = σ(w1 * x + b1)
    y = w2 * h + b2
    return y
end

# backpropagation
function backprop(x, y, learning_rate)
    h = σ(w1 * x + b1)           # 은닉층 출력 계산
    y_pred = w2 * h + b2         # 예측값 계산
    error = y - y_pred           # 오차 계산
    d_y_pred = error             # 출력층에서의 손실에 대한 미분값
    d_w2 = d_y_pred * h          # 출력층 가중치에 대한 그라디언트 계산
    d_b2 = d_y_pred              # 출력층 바이어스에 대한 그라디언트 계산
    d_h = d_y_pred * w2          # 은닉층 출력에 대한 그라디언트 전파
    d_w1 = d_h * x               # 은닉층 가중치에 대한 그라디언트 계산
    d_b1 = d_h                   # 은닉층 바이어스에 대한 그라디언트 계산

    # 가중치와 바이어스 업데이트
    w1 += learning_rate * d_w1
    b1 += learning_rate * d_b1
    w2 += learning_rate * d_w2
    b2 += learning_rate * d_b2
    return w1, b1, w2, b2
end

