require_relative 'activation_functions'
require_relative 'matrix_helpers'

module Qoa
  class NeuralNetwork
    include MatrixHelpers
    attr_reader :input_nodes, :hidden_nodes, :output_nodes, :learning_rate

    def initialize(input_nodes, hidden_nodes, output_nodes, learning_rate)
      @input_nodes = input_nodes
      @hidden_nodes = hidden_nodes
      @output_nodes = output_nodes
      @learning_rate = learning_rate

      @weights_ih = random_matrix(hidden_nodes, input_nodes)
      @weights_ho = random_matrix(output_nodes, hidden_nodes)
    end

    def random_matrix(rows, cols)
      Array.new(rows) { Array.new(cols) { rand * 2 - 1 } }
    end

    def train(inputs, targets)
      inputs = inputs.map { |x| [x] } # Convert to column vector
      targets = targets.map { |x| [x] } # Convert to column vector

      hidden_inputs = matrix_multiply(@weights_ih, inputs)
      hidden_outputs = apply_function(hidden_inputs, ActivationFunctions.method(:sigmoid))

      final_inputs = matrix_multiply(@weights_ho, hidden_outputs)
      final_outputs = apply_function(final_inputs, ActivationFunctions.method(:sigmoid))

      output_errors = matrix_subtract(targets, final_outputs)
      hidden_errors = matrix_multiply(transpose(@weights_ho), output_errors)

      gradients_ho = matrix_multiply_element_wise(output_errors, apply_function(final_outputs, ActivationFunctions.method(:sigmoid_derivative)))
      @weights_ho = matrix_add(@weights_ho, scalar_multiply(@learning_rate, matrix_multiply(gradients_ho, transpose(hidden_outputs))))

      gradients_ih = matrix_multiply_element_wise(hidden_errors, apply_function(hidden_outputs, ActivationFunctions.method(:sigmoid_derivative)))
      @weights_ih = matrix_add(@weights_ih, scalar_multiply(@learning_rate, matrix_multiply(gradients_ih, transpose(inputs))))
    end

    def query(inputs)
      inputs = inputs.map { |x| [x] } # Convert to column vector

      hidden_inputs = matrix_multiply(@weights_ih, inputs)
      hidden_outputs = apply_function(hidden_inputs, ActivationFunctions.method(:sigmoid))

      final_inputs = matrix_multiply(@weights_ho, hidden_outputs)
      final_outputs = apply_function(final_inputs, ActivationFunctions.method(:sigmoid))

      final_outputs.flatten
    end
  end
end