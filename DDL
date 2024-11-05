### Data Preparation
using Plots, JLD2

# Load Lorenz system data: Y ∈ ℝ^{3×T} where T is total time steps
@load "/Volumes/yoonJL/DeepApproximator/lorenz_data.jld2" t Y

# Partition data: input_data ∈ ℝ^T, target_data ∈ ℝ^{2×T}
input_data = Y[1, :]     # First component (x) of Lorenz system
target_data = Y[2:3, :]  # Second and third components (y,z)

### Data Segmentation
train_size = 100
# Training set: input_train ∈ ℝ^train_size, target_train ∈ ℝ^{2×train_size}
input_train = input_data[1:train_size]
target_train = target_data[:, 1:train_size]
# Testing set: remaining data
input_test = input_data[train_size+1:end]
target_test = target_data[:, train_size+1:end]

### Network Architecture Parameters
sequence_length = 100    # d: temporal sequence length
batch_size = 16         # Number of sequences per batch
epoch_length = 2000     # Total training iterations
hidden_size = 100       # h: number of hidden neurons
output_size = size(target_train, 1)  # q: output dimension (=2)
learning_rate = 0.01
num_samples = length(input_train) - sequence_length

### Parameter Initialization
# W_in ∈ ℝ^{h×d}: input-to-hidden weights
W_in = randn(hidden_size, sequence_length)
# W_out ∈ ℝ^{q×h}: hidden-to-output weights
W_out = randn(output_size, hidden_size)
# b_in ∈ ℝ^h: hidden layer bias
b_in = randn(hidden_size, 1)
# b_out ∈ ℝ^q: output layer bias
b_out = randn(output_size, 1)

### Training Process
for epoch in 1:epoch_length
    total_loss = 0
    num_batches = 0
    
    # Batch Processing
    for i in 1:batch_size:num_samples - sequence_length + 1
        num_batches += 1
        # x_batch ∈ ℝ^{d×n}: input sequences
        x_batch = zeros(sequence_length, batch_size)
        # y_batch ∈ ℝ^{q×n}: target outputs
        y_batch = zeros(output_size, batch_size)
        
        # Construct batch sequences
        for j in 0:batch_size - 1
            idx = i + j
            x_indices = idx : idx + sequence_length - 1
            x_batch[:, j+1] = input_train[x_indices]
            y_batch[:, j+1] = target_train[:, idx + sequence_length - 1]
        end

        ### Forward Propagation
        # h = σ(W_in X + b_in) ∈ ℝ^{h×n}
        h = tanh.(W_in * x_batch .+ b_in)
        # y_pred = W_out h + b_out ∈ ℝ^{q×n}
        y_pred = W_out * h .+ b_out

        ### Loss Computation
        loss = sum((y_batch .- y_pred).^2) / (batch_size * output_size)
        total_loss += loss

        ### Backward Propagation
        # Gradient computations preserving dimensionality
        dL_dy_pred = 2 .* (y_pred .- y_batch)    # ∈ ℝ^{q×n}
        dL_dW_out = (dL_dy_pred * h') / batch_size  # ∈ ℝ^{q×h}
        dL_db_out = sum(dL_dy_pred, dims=2) / batch_size  # ∈ ℝ^q
        dL_dh = W_out' * dL_dy_pred  # ∈ ℝ^{h×n}
        dL_dW_in = ((dL_dh .* (1 .- h .^ 2)) * x_batch') / batch_size  # ∈ ℝ^{h×d}
        dL_db_in = sum(dL_dh .* (1 .- h .^ 2), dims=2) / batch_size  # ∈ ℝ^h

        ### Parameter Updates
        W_in -= learning_rate .* dL_dW_in
        W_out -= learning_rate .* dL_dW_out
        b_in -= learning_rate .* dL_db_in
        b_out -= learning_rate .* dL_db_out

        # Early stopping criterion
        if total_loss / num_batches < 2e-3
            println("Early stopping at epoch $epoch")
            break
        end
    end
    println("Epoch: $epoch, Average Loss: $(total_loss / num_batches)")
end

### Prediction on Training Data
num_train_samples = sequence_length
# y_pred_train ∈ ℝ^{q×num_train_samples}
y_pred_train = zeros(output_size, num_train_samples)

for i in 1:num_train_samples
    x_indices = i : i + sequence_length - 1
    x_train = reshape(input_train[x_indices], sequence_length, 1)
    h_train = tanh.(W_in * x_train .+ b_in)
    y_pred_train[:, i] = W_out * h_train .+ b_out
end

### Visualization
plot(target_train[1, sequence_length:end], label="Actual (Training)")
plot!(y_pred_train[1, :], label="Predicted (Training)")
xlabel!("Time")
ylabel!("Value")
title!("Training Set Results")