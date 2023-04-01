require 'nmatrix/nmatrix'
require_relative 'activation_functions'

module Qoa
  class NeuralNetwork
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
      NMatrix.new([rows, cols], Array.new(rows * cols) { rand * 2 - 1 }, dtype: :float64)
    end

    def train(inputs, targets)
      inputs = NMatrix.new([inputs.size, 1], inputs)
      targets = NMatrix.new([targets.size, 1], targets)

      hidden_inputs = @weights_ih.transpose.dot(inputs)
      hidden_outputs = NMatrix.new([hidden_inputs.shape[0], 1], hidden_inputs.map { |x| ActivationFunctions.sigmoid(x) }.to_a.flatten, dtype: :float64)

      final_inputs = @weights_ho.transpose.dot(hidden_outputs)
      final_outputs = NMatrix.new([final_inputs.shape[0], 1], final_inputs.map { |x| ActivationFunctions.sigmoid(x) }.to_a.flatten, dtype: :float64)

      output_errors = targets - final_outputs
      hidden_errors = @weights_ho.dot(output_errors)

      @weights_ho += @learning_rate * (output_errors * final_outputs * (1.0 - final_outputs)).dot(hidden_outputs.transpose)
      @weights_ih += @learning_rate * (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)).dot(inputs.transpose)
    end

    def query(inputs)
      inputs = NMatrix.new([inputs.size, 1], inputs)
    
      hidden_inputs = @weights_ih.transpose.dot(inputs)
      hidden_outputs = NMatrix.new([hidden_inputs.shape[0], 1], hidden_inputs.map { |x| ActivationFunctions.sigmoid(x) }.to_a.flatten, dtype: :float64)

      final_inputs = @weights_ho.transpose.dot(hidden_outputs)
      final_outputs = final_inputs.map { |x| ActivationFunctions.sigmoid(x) }
    
      final_outputs.to_a.flatten
    end
  end
end