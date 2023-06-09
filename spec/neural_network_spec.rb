require 'spec_helper'

RSpec.describe Qoa::NeuralNetwork do
  let(:input_nodes) { 2 }
  let(:hidden_layers) { [[:conv, 32, 3], [:pool, 16, 2], 128, 64] }
  let(:output_nodes) { 1 }
  let(:learning_rate) { 0.1 }
  let(:activation_func) { :sigmoid }
  let(:dropout_rate) { 0.3 }
  let(:nn) { Qoa::NeuralNetwork.new(input_nodes, hidden_layers, output_nodes, learning_rate, dropout_rate) }

  context 'XOR problem' do
    it 'learns the XOR function' do
      xor_training_data = [
        { inputs: [0, 0], targets: [0] },
        { inputs: [0, 1], targets: [1] },
        { inputs: [1, 0], targets: [1] },
        { inputs: [1, 1], targets: [0] },
      ]
      50_000.times do
        data = xor_training_data.sample
        nn.train([data[:inputs]], [data[:targets]])
      end
      expect(nn.query([0, 0]).first.round).to be_within(0.2).of(0)
      expect(nn.query([0, 1]).first.round).to be_within(0.2).of(1)
      expect(nn.query([1, 0]).first.round).to be_within(0.2).of(1)
      expect(nn.query([1, 1]).first.round).to be_within(0.2).of(0)
    end
  end

  describe 'OR problem' do
    let(:or_training_data) do
      [
        { inputs: [0, 0], targets: [0] },
        { inputs: [0, 1], targets: [1] },
        { inputs: [1, 0], targets: [1] },
        { inputs: [1, 1], targets: [1] },
      ]
    end

    it 'learns the OR function' do
      50_000.times do
        data = or_training_data.sample
        nn.train([data[:inputs]], [data[:targets]])
      end

      expect(nn.query([0, 0]).first.round).to be_within(0.2).of(0)
      expect(nn.query([0, 1]).first.round).to be_within(0.2).of(1)
      expect(nn.query([1, 0]).first.round).to be_within(0.2).of(1)
      expect(nn.query([1, 1]).first.round).to be_within(0.2).of(1)
    end

    describe 'AND problem' do
      let(:and_training_data) do
        [
          { inputs: [0, 0], targets: [0] },
          { inputs: [0, 1], targets: [0] },
          { inputs: [1, 0], targets: [0] },
          { inputs: [1, 1], targets: [1] },
        ]
      end

      it 'learns the AND function' do
        50_000.times do
          data = and_training_data.sample
          nn.train([data[:inputs]], [data[:targets]])
        end

        expect(nn.query([0, 0]).first.round).to be_within(0.2).of(0)
        expect(nn.query([0, 1]).first.round).to be_within(0.2).of(0)
        expect(nn.query([1, 0]).first.round).to be_within(0.2).of(0)
        expect(nn.query([1, 1]).first.round).to be_within(0.2).of(1)
      end
    end
  end
end
