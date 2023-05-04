module Qoa
  module Layers
    class ConvolutionalLayer < Qoa::Layers::Layer
      attr_reader :kernel_size, :stride

      def initialize(input_size, output_size, kernel_size, stride = 1)
        super(input_size, output_size)
        @kernel_size = kernel_size
        @stride = stride
      end

      def random_matrix(rows, cols)
        limit = Math.sqrt(6.0 / (rows + cols))
        Array.new(rows) { Array.new(cols) { rand(-limit..limit) } }
      end
    end
  end
end
