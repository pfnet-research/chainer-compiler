#!/usr/bin/ruby

num_passing_node_tests = 0
File.readlines('scripts/runtests.py').each do |line|
  next if line =~ /^ *#/
  if line =~ /TestCase\(NODE_TEST, /
    num_passing_node_tests += 1
  end
end

num_node_tests = Dir.glob('onnx/onnx/backend/test/data/node/*').size

ops = []
File.readlines('onnx/docs/Operators.md').each do |line|
  if line =~ /^  \* <a href="#(.*?)"/
    ops << $1
  end
end

supported_ops = []
onikux_ops = []
File.readlines('compiler/gen_node.py').each do |line|
  if line !~ /^NodeDef\('(.*?)'/
    next
  end
  op = $1
  if op =~ /^Onikux/
    onikux_ops << op
  else
    supported_ops << op
  end
end

puts "Missing ops: #{ops - supported_ops}"

puts "Node tests: #{num_passing_node_tests}/#{num_node_tests}"
puts "Supported ops: #{supported_ops.size}/#{ops.size}"
puts "Custom ops: #{onikux_ops.size}"
