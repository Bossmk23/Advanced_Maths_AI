
# Stochastic Gradient Descent Made Simple: Learn by Building from Scratch

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

## Summary

Stochastic Gradient Descent (SGD) is a powerful optimization technique used in machine learning, especially for training large models efficiently. By updating model parameters using small batches of data, SGD minimizes computational cost and memory usage. It also introduces randomness, which helps escape local minima, improving the model's ability to generalize.

Through this tutorial, you’ve learned how SGD works by building an SGD optimizer from scratch and testing it on a simple binary classification problem. By applying these concepts, you can optimize machine learning models more effectively and tackle real-world problems in AI and data science. Mastering SGD opens up many opportunities in AI-driven industries.
