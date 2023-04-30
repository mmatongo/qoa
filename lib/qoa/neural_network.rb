require_relative 'layer'
require_relative 'activation_functions'
require_relative 'training'
require_relative 'utils'

module Qoa
  class NeuralNetwork
    include Training
    include Utils
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

      @layers = []
      @layers << Layer.new(input_nodes, hidden_layers[0])
      hidden_layers.each_cons(2) do |l1, l2|
        @layers << Layer.new(l1, l2)
      end
      @layers << Layer.new(hidden_layers[-1], output_nodes)
    end

    def query(inputs)
      layer_outputs = forward_pass(inputs)
      layer_outputs.last.flatten
    end

    def calculate_loss(inputs, targets)
      raise ArgumentError, 'inputs and targets must have the same length' if inputs.size != targets.size

      total_loss = 0.0
      inputs.zip(targets).each do |input, target|
        prediction = query(input)
        total_loss += mean_squared_error(prediction, target)
      end

      total_loss / inputs.size
    end

    def mean_squared_error(prediction, target)
      raise ArgumentError, 'prediction and target must have the same length' if prediction.size != target.size

      prediction.zip(target).map { |p, t| (p - t) ** 2 }.sum / prediction.size
    end
  end
end
