module Qoa
  module Layers
    class Layer
      attr_reader :input_size, :output_size, :weights

      def initialize(input_size, output_size)
        @input_size = input_size
        @output_size = output_size
        @weights = random_matrix(output_size, input_size)
      end

      def random_matrix(rows, cols)
        limit = Math.sqrt(6.0 / (rows + cols))
        Array.new(rows) { Array.new(cols) { rand(-limit..limit) } }
      end

      def weights=(new_weights)
        @weights = new_weights
      end
    end
  end
end
