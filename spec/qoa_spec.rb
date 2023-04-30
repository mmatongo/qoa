require 'spec_helper'

RSpec.describe Qoa::NeuralNetwork do
  let(:input_nodes) { 3 }
  let(:hidden_layers) { [4, 5] }
  let(:output_nodes) { 2 }
  let(:learning_rate) { 0.1 }
  let(:dropout_rate) { 0.1 }
  let(:activation_func) { :sigmoid }
  let(:neural_network) { Qoa::NeuralNetwork.new(input_nodes, hidden_layers, output_nodes, learning_rate, dropout_rate, activation_func) }

  describe '#initialize' do
    it 'creates a neural network with the specified parameters' do
      expect(neural_network.input_nodes).to eq(input_nodes)
      expect(neural_network.hidden_layers).to eq(hidden_layers)
      expect(neural_network.output_nodes).to eq(output_nodes)
      expect(neural_network.learning_rate).to eq(learning_rate)
    end
  end

  describe '#train' do
    let(:inputs) { [[0.5, 0.6, 0.7]] }
    let(:targets) { [[0.25, 0.75]] }

    it 'does not raise an error' do
      expect { neural_network.train(inputs, targets) }.not_to raise_error
    end
  end

  describe '#query' do
    let(:inputs) { [0.5, 0.6, 0.7] }

    it 'returns an array of output node values' do
      result = neural_network.query(inputs)
      expect(result).to be_an(Array)
      expect(result.size).to eq(output_nodes)
    end
  end

  describe 'network output changes after training' do
    let(:inputs) { [[0.5, 0.6, 0.7]] }
    let(:targets) { [[0.25, 0.75]] }

    it 'changes output after training' do
      initial_output = neural_network.query(inputs.first)
      neural_network.train(inputs, targets)
      new_output = neural_network.query(inputs.first)

      expect(new_output).not_to eq(initial_output)
    end
  end

  describe 'activation function application' do
    it 'applies the correct activation function' do
      inputs = [0.5, 0.6, 0.7]
      weights = neural_network.instance_variable_get(:@layers).map(&:weights)
      layer_outputs = [inputs.map { |x| [x] }]

      weights.each do |w|
        layer_inputs = neural_network.send(:matrix_multiply, w, layer_outputs.last)
        layer_outputs << neural_network.send(:apply_function, layer_inputs, Qoa::ActivationFunctions.method(activation_func))
      end

      layer_outputs.shift # Remove input layer
      layer_outputs.each.with_index do |layer_output, idx|
        expect(layer_output).to be_an(Array)
        expect(layer_output.size).to eq(hidden_layers[idx] || output_nodes)
      end
    end
  end

  describe 'weight matrices dimensions' do
    it 'has correct dimensions for the weight matrices' do
      weights = neural_network.instance_variable_get(:@layers).map(&:weights)

      expect(weights.first.size).to eq(hidden_layers.first)
      expect(weights.first.first.size).to eq(input_nodes)

      hidden_layers.each_cons(2) do |l1, l2|
        weight_matrix = weights[hidden_layers.index(l1) + 1]
        expect(weight_matrix.size).to eq(l2)
        expect(weight_matrix.first.size).to eq(l1)
      end

      expect(weights.last.size).to eq(output_nodes)
      expect(weights.last.first.size).to eq(hidden_layers.last)
    end
  end

  describe '#initialize' do
    it 'initializes with correct default values' do
      expect(neural_network.dropout_rate).to eq(dropout_rate)
      expect(neural_network.activation_func).to eq(activation_func)
    end
  end
end
