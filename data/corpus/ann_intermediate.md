## Page 1

# Artificial Neural Networks (ANN)

## Perceptron and Basic Architecture

Artificial neural networks are computational models inspired by biological neurons. The simplest unit is the perceptron, which takes multiple input features, multiplies them by learned weights, adds a bias term, and passes the result through an activation function to produce an output. When many perceptrons are stacked in layers — input, hidden, and output layers — the network can model complex nonlinear relationships. Each hidden layer transforms the representation learned from the previous layer, allowing the network to learn hierarchical feature abstractions. During training, weights are adjusted using gradient-based optimization methods to minimize prediction error. This layered structure enables neural networks to solve tasks such as regression, classification, and representation learning across many domains including vision, NLP, and tabular prediction.

## Activation Functions and Non-Linearity

Activation functions introduce non-linearity into neural networks, allowing them to learn complex decision boundaries. Without non-linear activations, stacking multiple layers would be equivalent to a single linear transformation. Common activation functions include ReLU, sigmoid, and tanh. ReLU is widely used because it mitigates the vanishing gradient problem and enables faster convergence during training. Sigmoid is useful in binary classification outputs because it maps values between 0 and 1, while tanh provides centered outputs between -1 and 1. The choice of activation function affects gradient flow, convergence speed, and overall model performance. Understanding activation behavior is crucial for designing stable and efficient deep neural network architectures.

---


## Page 2

## Backpropagation and Weight Optimization

Backpropagation is the core learning mechanism used to train artificial neural networks. It computes gradients of the loss function with respect to each weight using the chain rule of calculus. After a forward pass generates predictions, the loss function measures error between predictions and true labels. Gradients are then propagated backward through the network to update weights using optimization algorithms such as stochastic gradient descent or Adam. Learning rate selection plays a critical role: too high leads to divergence, while too low causes slow training. Backpropagation enables networks to iteratively improve performance by reducing prediction error, making it fundamental to supervised deep learning.

