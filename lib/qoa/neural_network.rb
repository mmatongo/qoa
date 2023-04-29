require 'concurrent'
require_relative 'activation_functions'
require_relative 'matrix_helpers'

module Qoa
  class NeuralNetwork
    include MatrixHelpers
    attr_reader :input_nodes, :hidden_layers, :output_nodes, :learning_rate, :activation_func, :dropout_rate, :decay_rate, :epsilon, :batch_size

    def initialize(input_nodes, hidden_layers, output_nodes, learning_rate, dropout_rate, activation_func = :sigmoid, decay_rate = 0.9, epsilon = 1e-8, batch_size = 10)
      @input_nodes = input_nodes
      @hidden_layers = hidden_layers
      @output_nodes = output_nodes
      @learning_rate = learning_rate
      @activation_func = activation_func
      @dropout_rate = dropout_rate
      @decay_rate = decay_rate
      @epsilon = epsilon
      @batch_size = batch_size

      @weights = []
      @weights << random_matrix(hidden_layers[0], input_nodes)
      hidden_layers.each_cons(2) do |l1, l2|
        @weights << random_matrix(l2, l1)
      end
      @weights << random_matrix(output_nodes, hidden_layers[-1])
    end

    def random_matrix(rows, cols)
      limit = Math.sqrt(6.0 / (rows + cols))
      Array.new(rows) { Array.new(cols) { rand(-limit..limit) } }
    end

    def train(inputs, targets)
      raise ArgumentError, 'inputs and targets must have the same length' if inputs.size != targets.size

      inputs.zip(targets).each_slice(@batch_size) do |batch|
        train_batch(batch)
      end
    end

    def train_batch(batch)
      derivative_func = "#{@activation_func}_derivative"
      batch_inputs = batch.map { |x, _| x }
      batch_targets = batch.map { |_, y| y }

      # Forward pass
      layer_outputs = batch_inputs.map { |inputs| forward_pass(inputs) }

      # Backward pass
      # Using thread pool to parallelize the backward pass for each input in the batch
      pool = Concurrent::FixedThreadPool.new(4)
      weight_deltas = Array.new(@weights.size) { Array.new(@weights[0].size) { Array.new(@weights[0][0].size, 0) } }
      mutex = Mutex.new

      batch.zip(layer_outputs).each do |(inputs, targets), outputs|
        pool.post do
          deltas = backward_pass(inputs, targets, outputs)
          mutex.synchronize do
            @weights.each_with_index do |_, i|
              weight_deltas[i] = matrix_add(weight_deltas[i], deltas[i])
            end
          end
        end
      end

      pool.shutdown
      pool.wait_for_termination

      # Update weights
      @weights.each_with_index do |w, i|
        @weights[i] = matrix_add(w, scalar_multiply(@learning_rate / batch.size, weight_deltas[i]))
      end
    end

    def forward_pass(inputs)
      inputs = inputs.map { |x| [x] } # Convert to column vector

      layer_outputs = [inputs]
      @weights.each_with_index do |w, i|
        layer_inputs = matrix_multiply(w, layer_outputs[-1])
        layer_outputs << apply_function(layer_inputs, ActivationFunctions.method(@activation_func))

        # Apply dropout to hidden layers
        layer_outputs[-1] = apply_dropout(layer_outputs[-1], @dropout_rate) if i < @weights.size - 1
      end

      layer_outputs
    end

    def backward_pass(inputs, targets, layer_outputs)
      derivative_func = "#{@activation_func}_derivative"
      inputs = inputs.map { |x| [x] } # Convert to column vector
      targets = targets.map { |x| [x] } # Convert to column vector

      # Compute errors
      errors = [matrix_subtract(targets, layer_outputs.last)]
      (@weights.size - 1).downto(1) do |i|
        errors << matrix_multiply(transpose(@weights[i]), errors.last)
      end

      # Compute weight deltas
      weight_deltas = []
      @weights.each_with_index do |w, i|
        gradients = matrix_multiply_element_wise(errors[i], apply_function(layer_outputs[i + 1], ActivationFunctions.method(derivative_func)))
        w_delta = matrix_multiply(gradients, transpose(layer_outputs[i]))
        weight_deltas << w_delta
      end

      weight_deltas
    end

    def query(inputs)
      layer_outputs = forward_pass(inputs)
      layer_outputs.last.flatten
    end
  end
end
