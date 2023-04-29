require_relative 'lib/qoa/version'

Gem::Specification.new do |s|
  s.name = 'qoa'
  s.version = Qoa::VERSION
  s.authors = ['Daniel M. Matongo']
  s.email = ['mmatongo_@hotmail.com']

  s.summary = 'A simple machine learning library'
  s.description = 'Qoa is a simple machine learning library for Ruby, including a basic feedforward neural network implementation with backpropagation.'
  s.homepage = 'https://github.com/mmatongo/qoa'
  s.license = 'Apache-2.0'

  s.files = Dir['{lib}/**/*', 'LICENSE', 'README.md', 'code_of_conduct.md']
  s.require_paths = ['lib']
  s.required_ruby_version = '>= 2.5.0'

  s.add_development_dependency 'bundler', '~> 2.0'
  s.add_development_dependency 'rake', '~> 13.0'
  s.add_development_dependency 'rspec', '~> 3.0'
end
