require_relative 'layers/layer'
require_relative 'layers/convolutional_layer'
require_relative 'layers/pooling_layer'
require_relative 'activation_functions'
require_relative 'training'
require_relative 'utils'
require_relative 'loss_functions'
require_relative 'err/validations'

module Qoa
  class NeuralNetwork
    include Training
    include Utils
    include LossFunctions
    include Err::Validations

    attr_reader :input_nodes, :hidden_layers, :output_nodes, :learning_rate, :activation_func, :dropout_rate, :decay_rate, :epsilon, :batch_size, :l1_lambda, :l2_lambda

    def initialize(input_nodes, hidden_layers, output_nodes, learning_rate, dropout_rate, activation_func = :relu, decay_rate = 0.9, epsilon = 1e-8, batch_size = 10, l1_lambda = 0.0, l2_lambda = 0.0)
      # validate_constructor_args(input_nodes, hidden_layers, output_nodes, learning_rate, dropout_rate, activation_func, decay_rate, epsilon, batch_size, l1_lambda, l2_lambda)

      @input_nodes = input_nodes
      @hidden_layers = hidden_layers
      @output_nodes = output_nodes
      @learning_rate = learning_rate
      @activation_func = activation_func
      @dropout_rate = dropout_rate
      @decay_rate = decay_rate
      @epsilon = epsilon
      @batch_size = batch_size
      @l1_lambda = l1_lambda
      @l2_lambda = l2_lambda

      @layers = []
      @layers << Qoa::Layers::Layer.new(input_nodes, hidden_layers[0].is_a?(Array) ? hidden_layers[0][1] : hidden_layers[0])

      hidden_layers.each_cons(2) do |l1, l2|
        l1_size = l1.is_a?(Array) ? l1[1] : l1
        l2_size = l2.is_a?(Array) ? l2[1] : l2

        if l1.is_a?(Array) && l2.is_a?(Array) && l1[0] == :conv && l2[0] == :conv
          @layers << Qoa::Layers::ConvolutionalLayer.new(l1_size, l2_size, l1[2], l1[3])
        elsif l1.is_a?(Array) && l1[0] == :conv && l2.is_a?(Numeric)
          @layers << Qoa::Layers::ConvolutionalLayer.new(l1_size, l2_size, l1[2], l1[3])
        elsif l1.is_a?(Numeric) && l2.is_a?(Array) && l2[0] == :conv
          @layers << Qoa::Layers::ConvolutionalLayer.new(l1_size, l2_size, l2[2], l2[3])
        elsif l1.is_a?(Array) && l1[0] == :pool && l2.is_a?(Numeric)
          @layers << Qoa::Layers::PoolingLayer.new(l1_size, l2_size, l1[2], l1[3])
        elsif l1.is_a?(Numeric) && l2.is_a?(Array) && l2[0] == :pool
          @layers << Qoa::Layers::PoolingLayer.new(l1_size, l2_size, l2[2], l2[3])
        else
          @layers << Qoa::Layers::Layer.new(l1_size, l2_size)
        end
      end
      @layers << Qoa::Layers::Layer.new(hidden_layers[-1].is_a?(Array) ? hidden_layers[-1][1] : hidden_layers[-1], output_nodes)
    end

    def query(inputs)
      validate_query_args(inputs)

      layer_outputs = forward_pass(inputs)
      layer_outputs.last.flatten
    end

    def calculate_loss(inputs, targets, loss_function = :cross_entropy_loss)
      validate_calculate_loss_args(inputs, targets, loss_function)

      total_loss = 0.0
      inputs.zip(targets).each do |input, target|
        prediction = query(input)
        total_loss += LossFunctions.send(loss_function, prediction, target)
      end

      total_loss / inputs.size
    end
  end
end
