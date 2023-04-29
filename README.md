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

### Option 1: Copy the files

Simply copy the `neural_network.rb`, `activation_functions.rb`, and `matrix_helpers.rb` files into your project and require them.

```ruby
require_relative 'neural_network'
require_relative 'activation_functions'
require_relative 'matrix_helpers'
```

### Option 2: Install via RubyGems

Alternatively, you can install the gem via RubyGems:

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
nn = Qoa::NeuralNetwork.new(784, [128, 64], 10, 0.001, 0.5, :relu, 0.9, 1e-8, 32)
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

The library is available as open source under the terms of the [MIT License](http://opensource.org/licenses/Apache-2.0).
