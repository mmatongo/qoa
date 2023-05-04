module Qoa
  module Layers
    class PoolingLayer < Qoa::Layers::Layer
      attr_reader :pool_size, :stride

      def initialize(input_size, output_size, pool_size, stride = 1)
        super(input_size, output_size)
        @pool_size = pool_size
        @stride = stride
      end
    end
  end
end
