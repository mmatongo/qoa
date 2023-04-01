require 'spec_helper'

describe Qoa::NeuralNetwork do
  let(:input_nodes) { 2 }
  let(:hidden_nodes) { 2 }
  let(:output_nodes) { 1 }
  let(:learning_rate) { 0.1 }
  let(:nn) { Qoa::NeuralNetwork.new(input_nodes, hidden_nodes, output_nodes, learning_rate) }

  context 'XOR problem' do
    it 'learns the XOR function' do
      training_data = [
        { inputs: [0, 0], targets: [0] },
        { inputs: [0, 1], targets: [1] },
        { inputs: [1, 0], targets: [1] },
        { inputs: [1, 1], targets: [0] }
      ]
      10000.times do
        data = training_data.sample
        nn.train(data[:inputs], data[:targets])
      end
      expect(nn.query([0, 0]).first.round).to eq(0)
      expect(nn.query([0, 1]).first.round).to eq(1)
      expect(nn.query([1, 0]).first.round).to eq(1)
      expect(nn.query([1, 1]).first.round).to eq(0)
    end
  end
end