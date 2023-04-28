require 'spec_helper'

RSpec.describe Qoa::NeuralNetwork do
  let(:input_nodes) { 3 }
  let(:hidden_nodes) { 4 }
  let(:output_nodes) { 2 }
  let(:learning_rate) { 0.1 }
  let(:neural_network) { Qoa::NeuralNetwork.new(input_nodes, hidden_nodes, output_nodes, learning_rate) }

  describe '#initialize' do
    it 'creates a neural network with the specified parameters' do
      expect(neural_network.input_nodes).to eq(input_nodes)
      expect(neural_network.hidden_nodes).to eq(hidden_nodes)
      expect(neural_network.output_nodes).to eq(output_nodes)
      expect(neural_network.learning_rate).to eq(learning_rate)
    end
  end

  describe '#train' do
    let(:inputs) { [0.5, 0.6, 0.7] }
    let(:targets) { [0.25, 0.75] }

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
end