module Qoa
  module ActivationFunctions
    class << self
      def sigmoid(x)
        1.0 / (1.0 + Math.exp(-x))
      end

      def self.sigmoid_derivative(x)
        x * (1.0 - x)
      end

      def tanh(x)
        Math.tanh(x)
      end

      def tanh_derivative(x)
        1.0 - (tanh(x) ** 2.0)
      end

      def relu(x)
        x < 0 ? 0 : x
      end

      def relu_derivative(x)
        x < 0 ? 0 : 1.0
      end
    end
  end
end