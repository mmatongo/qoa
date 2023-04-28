module Qoa
  module MatrixHelpers
    def matrix_multiply(a, b)
      a_rows = a.size
      a_cols = a[0].size
      b_rows = b.size
      b_cols = b[0].size
      raise ArgumentError, 'incompatible dimensions' if a_cols != b_rows

      result = Array.new(a_rows) { Array.new(b_cols, 0) }
      a_rows.times do |i|
        b_cols.times do |j|
          a_cols.times do |k|
            # Check for nil values before performing multiplication
            if a[i][k].nil? || b[k][j].nil?
              result[i][j] += 0
            else
              result[i][j] += a[i][k] * b[k][j]
            end
          end
        end
      end
      result
    end

    def apply_function(matrix, func)
      matrix.map { |row| row.map { |x| func.call(x) } }
    end

    def transpose(matrix)
      matrix[0].zip(*matrix[1..-1])
    end

    def matrix_subtract(a, b)
      a.zip(b).map { |r1, r2| r1.zip(r2).map { |x, y| x - y } }
    end

    def matrix_multiply_element_wise(a, b)
      a.zip(b).map do |r1, r2|
        r1.zip(r2).map do |x, y|
          x.nil? || y.nil? ? nil : x * y
        end
      end
    end

    def matrix_add(a, b)
      a.zip(b).map do |r1, r2|
        r1.zip(r2).map do |x, y|
          x.nil? || y.nil? ? nil : x + y
        end
      end
    end

    def scalar_multiply(scalar, matrix)
      matrix.map do |row|
        row.map do |x|
          x.nil? ? nil : x * scalar
        end
      end
    end

    def mean(matrix)
      matrix.map { |row| row.inject(0.0) { |sum, x| sum + x } / row.size }
    end

    def variance(matrix, mean)
      matrix.map.with_index { |row, i| row.inject(0.0) { |sum, x| sum + (x - mean[i]) ** 2 } / row.size }
    end

    def normalize(matrix, mean, variance)
      matrix.map.with_index { |row, i| row.map { |x| (x - mean[i]) / Math.sqrt(variance[i] + 1e-8) } }
    end

    def scale_and_shift(matrix, gamma, beta)
      matrix.map.with_index { |row, i| row.map { |x| gamma[i] * x + beta[i] } }
    end

    def update_gamma(gamma, normalized, gradients)
      gamma.each_with_index.map do |g, i|
        g + normalized[i].zip(gradients[i]).inject(0.0) { |sum, (n, d)| sum + n * d }
      end
    end

    def update_beta(beta, gradients)
      beta.each_with_index.map do |b, i|
        b + gradients[i].inject(0.0) { |sum, d| sum + d }
      end
    end

    def apply_dropout(matrix, dropout_rate)
      matrix.map { |row| row.map { |x| rand < dropout_rate ? 0 : x } }
    end

    def matrix_pow(matrix, power)
      matrix.map { |row| row.map { |x| x.nil? ? nil : x ** power } }
    end

    def scalar_add(matrix, scalar)
      matrix.map { |row| row.map { |x| x.nil? ? nil : x + scalar } }
    end
  end
end
