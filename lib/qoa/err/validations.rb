module Qoa
  module Err
    module Validations
      def validate_constructor_args(input_nodes, hidden_layers, output_nodes, learning_rate, dropout_rate, activation_func, decay_rate, epsilon, batch_size, l1_lambda, l2_lambda)
        raise ArgumentError, 'input_nodes, hidden_layers, and output_nodes must be positive integers' unless [input_nodes, output_nodes].all? { |x| x.is_a?(Integer) && x > 0 } && hidden_layers.is_a?(Array) && hidden_layers.all? { |x| x.is_a?(Integer) && x > 0 }
        raise ArgumentError, 'learning_rate, dropout_rate, decay_rate, epsilon, l1_lambda, and l2_lambda must be positive numbers' unless [learning_rate, dropout_rate, decay_rate, epsilon, l1_lambda, l2_lambda].all? { |x| x.is_a?(Numeric) && x >= 0 }
        raise ArgumentError, 'activation_func must be a valid symbol' unless ActivationFunctions.methods.include?(activation_func)
        raise ArgumentError, 'batch_size must be a positive integer' unless batch_size.is_a?(Integer) && batch_size > 0
      end

      def validate_query_args(inputs)
        raise ArgumentError, 'inputs must be an array of numbers' unless inputs.is_a?(Array) && inputs.all? { |x| x.is_a?(Numeric) }
      end

      def validate_calculate_loss_args(inputs, targets, loss_function)
        raise ArgumentError, 'inputs and targets must have the same length' if inputs.size != targets.size
        raise ArgumentError, 'inputs and targets must be arrays of arrays of numbers' unless inputs.is_a?(Array) && targets.is_a?(Array) && inputs.all? { |x| x.is_a?(Array) && x.all? { |y| y.is_a?(Numeric) } } && targets.all? { |x| x.is_a?(Array) && x.all? { |y| y.is_a?(Numeric) } }
        raise ArgumentError, 'loss_function must be a valid symbol' unless LossFunctions.methods.include?(loss_function)
      end

      def validate_train_args(inputs, targets)
        raise ArgumentError, 'inputs and targets must have the same length' if inputs.size != targets.size
        raise ArgumentError, 'inputs and targets must be arrays of arrays of numbers' unless inputs.is_a?(Array) && targets.is_a?(Array) && inputs.all? { |x| x.is_a?(Array) && x.all? { |y| y.is_a?(Numeric) } } && targets.all? { |x| x.is_a?(Array) && x.all? { |y| y.is_a?(Numeric) } }
      end
    end
  end
end
