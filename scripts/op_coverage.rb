#!/usr/bin/ruby

passing_node_tests = []
File.readlines('scripts/runtests.py').each do |line|
  next if line =~ /^ *#/
  if line =~ /TestCase\(NODE_TEST, '(.*?)'/
    passing_node_tests << $1
  end
end

node_tests = Dir.glob(
  'third_party/onnx/onnx/backend/test/data/node/*').map do |f|
  File.basename(f)
end

onnx_ops = []
File.readlines('third_party/onnx/docs/Operators.md').each do |line|
  if line =~ /^  \* <a href="#(.*?)"/
    onnx_ops << $1
  end
end

grad_ops = []
File.readlines('compiler/gradient_ops.cc').each do |line|
  if line =~ /^ *register_grad_fn\(Node::k(.*?), /
    grad_ops << $1
  end
end

ops = []
File.readlines('compiler/gen_node.py').each do |line|
  if line !~ /^NodeDef\('(.*?)'/
    next
  end
  ops << $1
end

def categorize(ops)
  supported_ops = []
  chainer_ops = []
  ops.each do |op|
    if op =~ /^Chainer/
      chainer_ops << op
    else
      supported_ops << op
    end
  end
  [supported_ops, chainer_ops]
end

supported_ops, chainer_ops = categorize(ops)
grad_onnx_ops, _ = categorize(grad_ops)

puts "Failing node tests:"
puts (node_tests - passing_node_tests).sort * "\n"
puts "Missing ops: #{onnx_ops - supported_ops}"
puts "Node tests: #{passing_node_tests.size}/#{node_tests.size}"
puts "Custom ops: #{chainer_ops.size}"
puts "Differentiable ops: #{grad_onnx_ops.size}"
puts "Supported ops: #{supported_ops.size}/#{onnx_ops.size}"
