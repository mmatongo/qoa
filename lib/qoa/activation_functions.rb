module Qoa
  module ActivationFunctions
    class << self
      def sigmoid(x)
        1.0 / (1.0 + Math.exp(-x))
      end

      def sigmoid_derivative(x)
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

      def leaky_relu(x, alpha = 0.01)
        x < 0 ? (alpha * x) : x
      end

      def leaky_relu_derivative(x, alpha = 0.01)
        x < 0 ? alpha : 1.0
      end

      def elu(x, alpha = 1.0)
        x < 0 ? (alpha * (Math.exp(x) - 1)) : x
      end

      def elu_derivative(x, alpha = 1.0)
        x < 0 ? (alpha * Math.exp(x)) : 1.0
      end

      def swish(x, beta = 1.0)
        x * sigmoid(beta * x)
      end

      def swish_derivative(x, beta = 1.0)
        swish(x, beta) + sigmoid(beta * x) * (1 - swish(x, beta))
      end

      def softmax(x)
        exps = x.map { |e| Math.exp(e - x.max) }
        sum = exps.inject(:+)
        exps.map { |e| e / sum }
      end

      def softmax_derivative(x)
        x.map { |e| e * (1 - e) }
      end
    end
  end
end
