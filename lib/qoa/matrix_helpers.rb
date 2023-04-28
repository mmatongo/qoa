module Qoa
  module MatrixHelpers
    def matrix_multiply(a, b)
      a_rows = a.size
      a_cols = a[0].size
      b_rows = b.size
      b_cols = b[0].size
      raise ArgumentError, "incompatible dimensions" if a_cols != b_rows

      result = Array.new(a_rows) { Array.new(b_cols, 0) }
      a_rows.times do |i|
        b_cols.times do |j|
          a_cols.times do |k|
            result[i][j] += a[i][k] * b[k][j]
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
      a.zip(b).map { |r1, r2| r1.zip(r2).map { |x, y| x * y } }
    end

    def matrix_add(a, b)
      a.zip(b).map { |r1, r2| r1.zip(r2).map { |x, y| x + y } }
    end

    def scalar_multiply(scalar, matrix)
      matrix.map { |row| row.map { |x| x * scalar } }
    end
  end
end
