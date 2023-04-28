require_relative 'activation_functions'
require_relative 'matrix_helpers'

module Qoa
  class NeuralNetwork
    include MatrixHelpers
    attr_reader :input_nodes, :hidden_nodes, :output_nodes, :learning_rate, :activation_func, :dropout_rate, :decay_rate, :epsilon

    def initialize(input_nodes, hidden_nodes, output_nodes, learning_rate, dropout_rate, activation_func = :sigmoid, decay_rate = 0.9, epsilon = 1e-8)
      @input_nodes = input_nodes
      @hidden_nodes = hidden_nodes
      @output_nodes = output_nodes
      @learning_rate = learning_rate
      @activation_func = activation_func
      @dropout_rate = dropout_rate
      @decay_rate = decay_rate
      @epsilon = epsilon

      @weights_ih = random_matrix(hidden_nodes, input_nodes)
      @weights_ho = random_matrix(output_nodes, hidden_nodes)

      # Initialize gamma and beta for batch normalization
      @gamma_ih = Array.new(hidden_nodes) { 1.0 }
      @beta_ih = Array.new(hidden_nodes) { 0.0 }
      @gamma_ho = Array.new(output_nodes) { 1.0 }
      @beta_ho = Array.new(output_nodes) { 0.0 }

      # Initialize cache for RMSprop
      @cache_ih = Array.new(hidden_nodes) { Array.new(input_nodes, 0) }
      @cache_ho = Array.new(output_nodes) { Array.new(hidden_nodes, 0) }
    end

    def random_matrix(rows, cols)
      Array.new(rows) { Array.new(cols) { rand * 2 - 1 } }
    end

    def train(inputs, targets)
      derivative_func = "#{@activation_func}_derivative"
      inputs = inputs.map { |x| [x] } # Convert to column vector
      targets = targets.map { |x| [x] } # Convert to column vector

      hidden_inputs = matrix_multiply(@weights_ih, inputs)
      hidden_outputs = apply_function(hidden_inputs, ActivationFunctions.method(@activation_func))

      # Apply dropout to hidden_outputs
      hidden_outputs = apply_dropout(hidden_outputs, @dropout_rate)

      final_inputs = matrix_multiply(@weights_ho, hidden_outputs)
      final_outputs = apply_function(final_inputs, ActivationFunctions.method(@activation_func))

      # Calculate mean and variance of the activations for batch normalization
      hidden_mean = mean(hidden_outputs)
      hidden_variance = variance(hidden_outputs, hidden_mean)
      final_mean = mean(final_outputs)
      final_variance = variance(final_outputs, final_mean)

      # Normalize the activations using mean and variance
      hidden_normalized = normalize(hidden_outputs, hidden_mean, hidden_variance)
      final_normalized = normalize(final_outputs, final_mean, final_variance)

      # Scale and shift the normalized activations using gamma and beta
      hidden_scaled_shifted = scale_and_shift(hidden_normalized, @gamma_ih, @beta_ih)
      final_scaled_shifted = scale_and_shift(final_normalized, @gamma_ho, @beta_ho)

      output_errors = matrix_subtract(targets, final_outputs)
      hidden_errors = matrix_multiply(transpose(@weights_ho), output_errors)

      gradients_ho = matrix_multiply_element_wise(output_errors, apply_function(final_outputs, ActivationFunctions.method(derivative_func)))
      gradients_ih = matrix_multiply_element_wise(hidden_errors, apply_function(hidden_outputs, ActivationFunctions.method(derivative_func)))

      # Update weights using RMSprop
      @cache_ih = matrix_add(scalar_multiply(@decay_rate, @cache_ih), scalar_multiply(1 - @decay_rate, matrix_pow(gradients_ih, 2)))
      @weights_ih = matrix_add(@weights_ih, scalar_multiply(@learning_rate, matrix_multiply_element_wise(matrix_pow(scalar_add(@cache_ih, @epsilon), -0.5), gradients_ih)))

      @cache_ho = matrix_add(scalar_multiply(@decay_rate, @cache_ho), scalar_multiply(1 - @decay_rate, matrix_pow(gradients_ho, 2)))
      @weights_ho = matrix_add(@weights_ho, scalar_multiply(@learning_rate, matrix_multiply_element_wise(matrix_pow(scalar_add(@cache_ho, @epsilon), -0.5), gradients_ho)))

      # Update gamma and beta for hidden and output layers
      @gamma_ih = update_gamma(@gamma_ih, hidden_normalized, gradients_ih)
      @beta_ih = update_beta(@beta_ih, gradients_ih)
      @gamma_ho = update_gamma(@gamma_ho, final_normalized, gradients_ho)
      @beta_ho = update_beta(@beta_ho, gradients_ho)
    end

    def query(inputs)
      inputs = inputs.map { |x| [x] } # Convert to column vector

      hidden_inputs = matrix_multiply(@weights_ih, inputs)
      hidden_outputs = apply_function(hidden_inputs, ActivationFunctions.method(@activation_func))

      # Apply dropout to hidden_outputs
      hidden_outputs = apply_dropout(hidden_outputs, @dropout_rate)

      final_inputs = matrix_multiply(@weights_ho, hidden_outputs)
      final_outputs = apply_function(final_inputs, ActivationFunctions.method(@activation_func))

      final_outputs.flatten
    end
  end
end
