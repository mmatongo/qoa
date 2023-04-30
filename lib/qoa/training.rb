require_relative 'matrix_helpers'
require 'concurrent'

module Qoa
  module Training
    include MatrixHelpers

    def train(inputs, targets)
      raise ArgumentError, 'inputs and targets must have the same length' if inputs.size != targets.size

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
        layer.weights = matrix_add(layer.weights, scalar_multiply(@learning_rate / batch.size, weight_deltas[i]))
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
      @layers.map(&:weights).each_with_index do |w, i|
        layer_inputs = matrix_multiply(w, layer_outputs[-1])
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
      @layers.each_with_index do |_, i|
        gradients = matrix_multiply_element_wise(errors[i], apply_function(layer_outputs[i + 1], ActivationFunctions.method(derivative_func)))
        w_delta = matrix_multiply(gradients, transpose(layer_outputs[i]))
        weight_deltas << w_delta
      end

      weight_deltas
    end
  end
end
