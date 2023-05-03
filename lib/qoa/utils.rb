require 'json'

module Qoa
  module Utils
    def save_model(file_path)
      model_data = {
        input_nodes: @input_nodes,
        hidden_layers: @hidden_layers,
        output_nodes: @output_nodes,
        learning_rate: @learning_rate,
        activation_func: @activation_func,
        dropout_rate: @dropout_rate,
        decay_rate: @decay_rate,
        epsilon: @epsilon,
        batch_size: @batch_size,
        weights: @layers.map(&:weights),
      }

      File.open(file_path, 'w') do |f|
        f.write(JSON.pretty_generate(model_data))
      end
    end

    def load_model(file_path)
      model_data = JSON.parse(File.read(file_path), symbolize_names: true)

      @input_nodes = model_data[:input_nodes]
      @hidden_layers = model_data[:hidden_layers]
      @output_nodes = model_data[:output_nodes]
      @learning_rate = model_data[:learning_rate]
      @activation_func = model_data[:activation_func].to_sym
      @dropout_rate = model_data[:dropout_rate]
      @decay_rate = model_data[:decay_rate]
      @epsilon = model_data[:epsilon]
      @batch_size = model_data[:batch_size]

      @layers = model_data[:weights].map { |w| Layer.new(w.first.size, w.size) }
      @layers.each_with_index do |layer, i|
        layer.weights = model_data[:weights][i]
      end
    end
  end
end
