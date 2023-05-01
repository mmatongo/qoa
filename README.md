# Qoa Neural Network

Qoa is a simple and customizable neural network library for Ruby. It allows you to create, train, and evaluate feedforward neural networks with various activation functions, dropout rates, and learning parameters.

## Features

- Supports various activation functions
- Dropout regularization
- Mini-batch gradient descent
- Weight initialization using Xavier initialization
- Customizable learning parameters
- Parallelized backward pass for faster training

## Installation

### Install via RubyGems

You can install the gem via RubyGems:

```
gem install qoa
```

Then, require the gem in your project:

```ruby
require 'qoa'
```

## Usage

### Creating a Neural Network

To create a new neural network, you can initialize an instance of `Qoa::NeuralNetwork` with the following parameters:

- `input_nodes`: The number of input nodes.
- `hidden_layers`: An array of the number of nodes in each hidden layer.
- `output_nodes`: The number of output nodes.
- `learning_rate`: The learning rate for the gradient descent optimization.
- `dropout_rate`: The dropout rate for regularization.
- `activation_func`: The activation function to use (default is `:sigmoid`).
- `decay_rate`: The decay rate for the RMSProp optimizer (default is `0.9`).
- `epsilon`: A small value to prevent division by zero in the RMSProp optimizer (default is `1e-8`).
- `batch_size`: The size of the mini-batches used for training (default is `10`).

Example:

```ruby
require 'qoa'

input_nodes = 784 # Number of input features (e.g., 28x28 pixels for MNIST dataset)
hidden_layers = [128, 64] # Two hidden layers with 128 and 64 nodes each
output_nodes = 10 # Number of output classes (e.g., 10 for MNIST dataset)
learning_rate = 0.01
dropout_rate = 0.5
activation_func = :relu

nn = Qoa::NeuralNetwork.new(input_nodes, hidden_layers, output_nodes, learning_rate, dropout_rate, activation_func)
```

### Saving and Loading Models

To save the trained model to a file, call the `save_model` method:

```ruby
nn.save_model('model.json')
```

To load a previously saved model, call the `load_model` method:

```ruby
nn.load_model('model.json')
```

### Training the Neural Network

To train the neural network, you can call the `train` method with the following parameters:

- `inputs`: An array of input vectors (each vector should have the same length as the number of input nodes).
- `targets`: An array of target vectors (each vector should have the same length as the number of output nodes).

Example:

```ruby
inputs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
nn.train(inputs, targets)
```

### Evaluating the Neural Network

To evaluate the neural network, you can call the `query` method with the following parameter:

- `inputs`: An input vector (should have the same length as the number of input nodes).

The method will return an output vector with the same length as the number of output nodes.

Example:

```ruby
inputs = [0.1, 0.2, 0.3]
output = nn.query(inputs)
puts output.inspect
```

## Activation Functions

The library supports the following activation functions:

- Sigmoid (default)
- Hyperbolic tangent (tanh)
- Rectified linear unit (ReLU)
- Leaky rectified linear unit (Leaky ReLU)
- Exponential linear unit (ELU)
- Swish
- Softmax

To use a different activation function, simply pass its symbol when creating a new neural network. For example:

```ruby
nn = Qoa::NeuralNetwork.new(784, [128, 64], 10, 0.001, 0.5, :tanh)
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/mmatongo/qoa. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The library is available as open source under the terms of the [Apache-2.0 License](http://opensource.org/licenses/Apache-2.0).
