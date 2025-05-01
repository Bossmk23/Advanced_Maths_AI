
# Cracking the Code of Optimization: A Comprehensive Guide to SGD, RMSprop, and Adam

# Introduction of SGD

Stochastic Gradient Descent (SGD) is one of the most widely used optimization techniques in machine learning. It is especially important in training models such as linear regression, logistic regression, and deep neural networks. SGD helps models learn from data by gradually improving predictions through repeated updates to model parameters.

## What is Optimization?

Optimization in machine learning refers to the process of minimizing a loss function to improve model performance. The loss function measures how far off the model's predictions are from the actual values. By minimizing this loss, the model becomes more accurate.

## Advantages of SGD

Here are the key advantages of Stochastic Gradient Descent (SGD):

- **Faster and Memory-Efficient:** Since SGD updates model parameters using small batches instead of the full dataset, it significantly reduces computation time and memory usage, making it ideal for large-scale data.

- **Helps Escape Local Minima:** The randomness in data batches allows SGD to jump out of local minima, helping it find better overall solutions.

- **Simple and Effective:** SGD is easy to implement and supported by most machine learning libraries. Its use of mini-batches and randomness not only reduces computational load but also helps the model avoid getting stuck in poor solutions, improving its ability to generalize on new data.

- **Supports Online Learning:** SGD can update the model as new data arrives, making it suitable for real-time applications.

## Working of SGD (Theory)

Stochastic Gradient Descent works by repeatedly updating the model's parameters in small steps using random mini-batches of the data. This approach makes training faster and allows for continuous updates, which is useful in streaming or online settings.

At each iteration, SGD:

- Picks a random batch of training samples.
- Calculates the loss and gradient.
- Updates weights and bias to reduce the loss.
- Repeats the above steps for several epochs (full passes through the data).

By doing this many times, the model gradually reduces its loss and improves its predictions. The final result is a trained model that has learned from the data and is better at making accurate predictions.

## Key Terms to Remember

- **Epoch:** One full pass through the entire dataset. It defines how many times the learning algorithm will work through the entire training dataset.

- **Batch Size:** The number of training examples used in one update step.

- **Learning Rate (η):** A small number that controls how big each update step is.

- **Formula:** θ = θ - η * ∇J(θ) where θ represents the parameters, η is the learning rate, and ∇J(θ) is the gradient.

- **Gradient:** The direction and rate of steepest increase in loss; we move in the opposite direction to reduce the loss.

- **For logistic regression:** ∇w = (1/m) * Xᵀ * (ŷ - y).

- **Sigmoid Function:** Used for binary classification to squash values between 0 and 1.

- **Formula:** σ(z) = 1 / (1 + e^(-z)).

- **Binary Cross-Entropy:** A loss function used for binary classification problems.

- **Formula:** L = - (1/m) * Σ [y * log(ŷ) + (1 - y) * log(1 - ŷ)].

## Code Demonstration: Implementing SGD

Let's implement SGD from scratch to better understand how it works. We will create a simple binary classification dataset and use SGD to optimize the parameters of a logistic regression model.

```python
import numpy as np
import matplotlib.pyplot as plt

# ---------- Step 1: Generate a Simple Binary Dataset ----------
# Create 2D data (x1, x2) and binary labels (0 or 1)
def create_dataset(n_samples=100):
    np.random.seed(42)
    # Class 0: Centered at (1,1)
    class_0 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
    # Class 1: Centered at (3,3)
    class_1 = np.random.randn(n_samples // 2, 2) + np.array([3, 3])

    # Combine data and create labels
    X = np.vstack((class_0, class_1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    return X, y

# ---------- Step 2: Define Sigmoid and Loss Function ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    # Clip values to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ---------- Step 3: Modular SGD Function ----------
def sgd_optimizer(X, y, learning_rate=0.1, epochs=50, batch_size=16, shuffle=True):
    n_samples, n_features = X.shape

    # Initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0

    # For tracking loss
    loss_history = []

    for epoch in range(epochs):
        # Shuffle data if needed
        if shuffle:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

        # Mini-batch training
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            # Predictions
            linear_output = np.dot(X_batch, weights) + bias
            predictions = sigmoid(linear_output)

            # Gradients
            error = predictions - y_batch
            dw = np.dot(X_batch.T, error) / len(y_batch)
            db = np.mean(error)

            # Update parameters
            weights -= learning_rate * dw
            bias -= learning_rate * db

        # Track and print loss per epoch
        final_preds = sigmoid(np.dot(X, weights) + bias)
        loss = binary_cross_entropy(y, final_preds)
        loss_history.append(loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    return weights, bias, loss_history

# ---------- Step 4: Run Everything ----------
# Generate data
X, y = create_dataset()

# Train using our SGD optimizer
weights, bias, losses = sgd_optimizer(X, y, learning_rate=0.1, epochs=50, batch_size=16)

# ---------- Step 5: Plot Loss vs Epoch ----------
plt.figure(figsize=(8, 5))
plt.plot(losses, marker='o')
plt.title("Loss vs Epochs (SGD on Logistic Regression)")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()
```

## Output

When running this code, the following output will be printed for each epoch:

```yaml
Epoch 1/50, Loss: 0.6886
Epoch 2/50, Loss: 0.6813
Epoch 3/50, Loss: 0.6742
...
Epoch 48/50, Loss: 0.5971
Epoch 49/50, Loss: 0.5947
Epoch 50/50, Loss: 0.5923
```

Additionally, a plot showing Loss vs Epoch will appear. The loss decreases over time, indicating that the SGD optimizer is working to reduce the binary cross-entropy and improve model accuracy.

# Introduction of RMSprop

RMSprop (Root Mean Square Propagation) is an optimization algorithm commonly used for training machine learning models, particularly deep neural networks. It improves the learning process by dynamically adjusting the learning rate for each parameter based on recent gradient magnitudes. Unlike basic methods like SGD, which use a fixed step size, RMSprop adapts the step size during training, allowing for more efficient and stable convergence. This helps avoid issues such as overshooting the minimum or slowing down unnecessarily. RMSprop is especially effective in handling non-stationary or noisy data and performs well in models like recurrent neural networks (RNNs).

## Advantages of RMSprop

Some advantages of using RMSprop optimization technique are:  

- **Adaptive Learning Rates:** It automatically adjusts the learning rate for each parameter, making training more efficient and reducing the need for manual tuning.  
- **Works Well with Noisy Data:** RMSprop performs better than SGD on noisy problems, such as online learning or mini-batch updates.  
- **Stable and Faster Convergence:** It helps models converge more quickly and smoothly by avoiding the zig-zagging often seen in standard SGD.  
- **Good for RNNs:** RMSprop is particularly effective for training recurrent neural networks, where gradients can vary widely in scale.  
- **Less Sensitive to Initial Learning Rate:** Since it adapts the step size, RMSprop is often more forgiving when the initial learning rate isn’t perfect.

## Working of RMSprop (THEORY)

- **Starts like SGD:** RMSprop also uses gradients to update model parameters and tries to minimize the loss — just like SGD.  
- **Tracks Past Gradients:** But instead of blindly using the current gradient, RMSprop keeps a moving average of the squared gradients for each parameter.  
- **Slows Down Big Updates:** If a parameter’s gradient has been large for a while, RMSprop reduces the learning rate for that parameter — so it doesn’t overshoot.  
- **Speeds Up Small Updates:** If a parameter’s gradient is small, it increases the learning rate slightly — helping it to keep learning and not get stuck.  
- **Adapts Learning Rate Individually:** Unlike SGD which uses the same learning rate for all parameters, RMSprop adapts the learning rate individually for each one — making training more stable.  
- **Faster & Smoother Convergence:** This technique leads to faster and more reliable training, especially when dealing with complex models like RNNs or data with noisy gradients.

## Code: Using RMSprop for Logistic Regression

This code demonstrates the use of the **RMSprop** optimizer in a simple logistic regression model. It shows how to compile and train the model using **RMSprop** to optimize the learning rate during training.

```python
# ---------- Step 2: Create a Logistic Regression Model ----------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,), activation='sigmoid')  # Logistic regression
])

# ---------- Step 3: Compile the Model with RMSprop ----------
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),  # Using RMSprop optimizer
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------- Step 4: Train the Model ----------
history = model.fit(X, y, epochs=50, batch_size=16, verbose=1, shuffle=True)
```

# Introduction of Adam

Adam (Adaptive Moment Estimation) is an advanced optimization algorithm commonly used in training machine learning and deep learning models. It combines the benefits of Momentum (which smooths updates) and RMSprop (which adapts learning rates per parameter). Adam tracks both the mean and variance of past gradients to make more adaptive and efficient updates. This leads to faster and more stable convergence during training. It’s widely preferred in machine learning tasks due to its robustness on noisy data and minimal need for tuning.

## Advantages of Adam

- **Adaptive Learning Rates:** Adam automatically adjusts the learning rate for each parameter, which improves training speed and efficiency.  
- **Combines Momentum & RMSprop:** It blends the benefits of momentum (smooth updates) and RMSprop (adaptive learning rates), making it powerful and balanced.  
- **Works Well with Sparse Gradients:** Adam performs well on problems with sparse or noisy gradients, such as text or natural language processing tasks.  
- **Requires Minimal Tuning:** Default hyperparameters often work well, so you don't need to spend much time tuning.  
- **Fast Convergence:** It generally converges faster than SGD and other optimizers, especially in complex or large-scale models.  
- **Widely Supported:** Adam is available in all major machine learning libraries like TensorFlow, PyTorch, and Keras.

## Working of Adam (THEORY)

Adam stands for Adaptive Moment Estimation. It combines the best parts of Momentum and RMSprop to make learning faster and smarter.

- **How do we start:**  
We start with initialization of two moving averages for each parameter:  
1. `m` for the mean of gradients (momentum).  
2. `v` for the squared gradients (like in RMSprop).  
These are both set to zero at the start.

- **Gradient Calculation:**  
For each parameter, Adam computes the gradient of the loss function during training (just like SGD).

- **Update Moving Averages:**  
1. `m` is updated using the current gradient to track the direction of the gradient.  
2. `v` is updated using the square of the gradient to track its magnitude.

- **Bias Correction:**  
Each parameter is updated by combining `m` and `v` — the update is scaled by `m / (√v + ε)`, which gives an adaptive learning rate for each parameter.

- **Repeat the Process:**  
This cycle is repeated for each batch/epoch until the model converges to the optimal solution.

## Code: Using RMSprop

This code shows how the **Adam optimizer** is applied to a simple linear regression model using PyTorch. It updates weights adaptively for better convergence by combining momentum and RMSprop principles.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert data to torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# Define simple linear model
model = nn.Linear(X_train.shape[1], 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
adam_losses = []
epochs = 200

for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    
    adam_losses.append(loss.item())
```

### Optimizer Comparison: SGD vs RMSprop vs Adam

| Feature / Optimizer       | **SGD**                                     | **RMSprop**                                        | **Adam**                                                  |
|--------------------------|---------------------------------------------|---------------------------------------------------|-----------------------------------------------------------|
| **Learning Rate**         | Fixed; needs tuning                        | Adaptive to recent gradients                      | Adaptive using both gradient mean & variance              |
| **Gradient Use**          | Uses current gradient only                 | Uses moving average of squared gradients          | Uses moving average of both gradients & squared gradients |
| **Stability & Speed**     | Can be unstable and slow                   | More stable, faster than SGD                      | Highly stable, fast convergence                           |
| **Best Use Cases**        | Simple models, basic tasks                 | Noisy data, RNNs                                  | Large models, NLP, CV, deep learning                      |
| **Applications**          | Image classification, shallow nets         | Online learning, sequential data                  | NLP, computer vision, large datasets                      |
| **Tuning Needs**          | High                                        | Moderate                                          | Low (works well with defaults)                            |
| **Library Support**       | Available in TensorFlow & PyTorch          | Available in TensorFlow & PyTorch                 | Widely supported across all ML libraries                  |
| **Beginner Friendly**     | Best for learning basics                   | Intermediate                                      | Yes; commonly used as default                            |

## Conclusion

Nowadays, when machine learning is trending, we need to update ourselves by learning optimization techniques like **SGD**, **RMSprop**, and **Adam** to enhance model training efficiency.  
**SGD (Stochastic Gradient Descent)** is the most basic optimizer, ideal for understanding foundational learning but often slow and unstable.  
**RMSprop** improves upon SGD by adapting learning rates for each parameter using recent gradient information, making it effective for noisy data and RNNs.  
**Adam** combines the strengths of Momentum and RMSprop, offering fast and stable convergence, especially useful in deep learning, NLP, and computer vision.  
These optimizers play a key role in how models learn — by controlling learning rates, using gradient history, and ensuring convergence stability.  
The right choice depends on the task, data type, and complexity of the model being built.  **In a fast-moving field like machine learning, continuous learning and timely adaptation aren't optional — they're essential for staying ahead and building smarter solutions.**

