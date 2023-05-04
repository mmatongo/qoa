require 'concurrent'
require_relative 'matrix_helpers'
require_relative 'err/validations'

module Qoa
  module Training
    include MatrixHelpers
    include Err::Validations

    def train(inputs, targets)
      validate_train_args(inputs, targets)

      inputs.zip(targets).each_slice(@batch_size) do |batch|
        train_batch(batch)
      end
    end

    def train_batch(batch)
      derivative_func = "#{@activation_func}_derivative"
      batch_inputs = batch.map { |x, _| x }
      batch_targets = batch.map { |_, y| y }

      # Forward pass
      layer_outputs = batch_inputs.map { |inputs| forward_pass(inputs) }

      # Backward pass
      # Using thread pool to parallelize the backward pass for each input in the batch
      pool = Concurrent::FixedThreadPool.new(4)
      #   weight_deltas = Array.new(@layers.size - 1) { |i| Array.new(@layers[i].output_size) { Array.new(@layers[i].input_size, 0) } }
      weight_deltas = Array.new(@layers.size) { |i| Array.new(@layers[i].output_size) { Array.new(@layers[i].input_size, 0) } }
      mutex = Mutex.new

      batch.zip(layer_outputs).each do |(inputs, targets), outputs|
        pool.post do
          deltas = backward_pass(inputs, targets, outputs)
          mutex.synchronize do
            @layers.each_with_index do |_, i|
              weight_deltas[i] = matrix_add(weight_deltas[i], deltas[i])
            end
          end
        end
      end

      pool.shutdown
      pool.wait_for_termination

      # Update weights
      @layers.each_with_index do |layer, i|
        regularization_penalty = calculate_regularization_penalty(layer.weights, @l1_lambda, @l2_lambda)
        layer.weights = matrix_add(layer.weights, scalar_multiply(@learning_rate / batch.size, matrix_add(weight_deltas[i], regularization_penalty)))
      end
    end

    def train_with_early_stopping(inputs, targets, validation_inputs, validation_targets, max_epochs, patience)
      best_validation_loss = Float::INFINITY
      patience_left = patience
      epoch = 0

      while epoch < max_epochs && patience_left > 0
        train(inputs, targets)
        validation_loss = calculate_loss(validation_inputs, validation_targets)
        puts "Epoch #{epoch + 1}: Validation loss = #{validation_loss}"

        if validation_loss < best_validation_loss
          best_validation_loss = validation_loss
          save_model('best_model.json')
          patience_left = patience
        else
          patience_left -= 1
        end

        epoch += 1
      end

      puts "Training stopped. Best validation loss = #{best_validation_loss}"
      load_model('best_model.json')
    end

    def forward_pass(inputs)
      inputs = inputs.map { |x| [x] } # Convert to column vector

      layer_outputs = [inputs]
      @layers.each_with_index do |layer, i|
        if layer.is_a?(Qoa::Layers::ConvolutionalLayer)
          layer_inputs = convolution(layer, layer_outputs[-1])
        elsif layer.is_a?(Qoa::Layers::PoolingLayer)
          layer_inputs = pooling(layer, layer_outputs[-1])
        else
          layer_inputs = matrix_multiply(layer.weights, layer_outputs[-1])
        end

        layer_outputs << apply_function(layer_inputs, ActivationFunctions.method(@activation_func))

        # Apply dropout to hidden layers
        layer_outputs[-1] = apply_dropout(layer_outputs[-1], @dropout_rate) if i < @layers.size - 2
      end

      layer_outputs
    end

    def backward_pass(inputs, targets, layer_outputs)
      derivative_func = "#{@activation_func}_derivative"
      inputs = inputs.map { |x| [x] } # Convert to column vector
      targets = targets.map { |x| [x] } # Convert to column vector

      # Compute errors
      errors = [matrix_subtract(targets, layer_outputs.last)]
      (@layers.size - 2).downto(0) do |i|
        errors << matrix_multiply(transpose(@layers[i + 1].weights), errors.last)
      end

      # Compute weight deltas
      weight_deltas = []
      @layers.each_with_index do |layer, i|
        gradients = matrix_multiply_element_wise(errors[i], apply_function(layer_outputs[i + 1], ActivationFunctions.method(derivative_func)))
        if layer.is_a?(Qoa::Layers::ConvolutionalLayer)
          w_delta = conv_weight_delta(layer, gradients, layer_outputs[i])
        elsif layer.is_a?(Qoa::Layers::PoolingLayer)
          w_delta = pool_weight_delta(layer, gradients, layer_outputs[i])
        else
          w_delta = matrix_multiply(gradients, transpose(layer_outputs[i]))
        end
        weight_deltas << w_delta
      end

      weight_deltas
    end

    def calculate_regularization_penalty(weights, l1_lambda, l2_lambda)
      l1_penalty = weights.map do |row|
        row.nil? ? nil : row.map { |x| x.nil? ? nil : (x < 0 ? -1 : 1) }
      end
      l1_penalty = scalar_multiply(l1_lambda, l1_penalty)

      l2_penalty = scalar_multiply(l2_lambda, weights)

      matrix_add(l1_penalty, l2_penalty)
    end

    def convolution(layer, inputs)
      output_size = layer.output_size
      kernel_size = layer.kernel_size
      stride = layer.stride

      output = Array.new(output_size) { Array.new(inputs.length - kernel_size + 1) }
      layer.weights.each_with_index do |row, i|
        inputs.each_cons(kernel_size).each_with_index do |input_slice, j|
          output[i][j] = row.zip(input_slice).map { |a, b| a * b }.reduce(:+)
        end
      end

      output
    end

    def pooling(layer, inputs)
      output_size = layer.output_size
      pool_size = layer.pool_size
      stride = layer.stride || 1

      # Calculate the number of columns in the output array
      output_columns = inputs[0].length - pool_size
      output_columns = output_columns <= 0 ? 1 : ((output_columns) / stride.to_f).ceil + 1

      output = Array.new(output_size) { Array.new(output_columns) }

      (0...output_size).each do |i|
        (0...output_columns).each do |j|
          start_idx = j * stride
          end_idx = start_idx + pool_size - 1
          next if inputs[i].nil? || inputs[i].length < end_idx + 1 # Add this check to avoid accessing an index that does not exist
          pool_slice = inputs[i].slice(start_idx..end_idx)
          output[i][j] = pool_slice.max
        end
      end

      output
    end

    def conv_weight_delta(layer, gradients, inputs)
      kernel_size = layer.kernel_size
      stride = layer.stride

      deltas = layer.weights.map do |row|
        inputs.each_cons(kernel_size).map do |input_slice|
          row.zip(input_slice).map { |a, b| a * b }.reduce(:+)
        end
      end

      deltas
    end

    def pool_weight_delta(layer, gradients, inputs)
      pool_size = layer.pool_size
      stride = layer.stride

      deltas = inputs.each_slice(stride).map do |input_slice|
        input_slice.each_cons(pool_size).map do |pool_slice|
          max_index = pool_slice.each_with_index.max[1]
          pool_slice.map.with_index { |v, i| (i == max_index) ? v * gradients[i] : 0 }
        end
      end

      deltas
    end
  end
end
