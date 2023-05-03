module Qoa
  module LossFunctions
    class << self
      def mean_squared_error(prediction, target)
        raise ArgumentError, 'prediction and target must have the same length' if prediction.size != target.size
        prediction.zip(target).map { |p, t| (p - t) ** 2 }.sum / prediction.size
      end

      def cross_entropy_loss(prediction, target)
        raise ArgumentError, 'prediction and target must have the same length' if prediction.size != target.size
        -prediction.zip(target).map { |p, t| t * Math.log(p) }.sum / prediction.size
      end

      def binary_cross_entropy(prediction, target)
        raise ArgumentError, 'prediction and target must have the same length' if prediction.size != target.size
        -prediction.zip(target).map { |p, t| t * Math.log(p) + (1 - t) * Math.log(1 - p) }.sum / prediction.size
      end

      def categorical_cross_entropy(prediction, target)
        raise ArgumentError, 'prediction and target must have the same length' if prediction.size != target.size
        -prediction.zip(target).map { |p, t| t * Math.log(p) }.sum / prediction.size
      end

      def mean_absolute_error(prediction, target)
        raise ArgumentError, 'prediction and target must have the same length' if prediction.size != target.size
        prediction.zip(target).map { |p, t| (p - t).abs }.sum / prediction.size
      end
    end
  end
end
