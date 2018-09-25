#!/usr/bin/ruby
#
# Usage:
#
# $ ./scripts/runtests.py -g onnx_real --show_log |& tee log
# $ ./scripts/parse_elapsed.rb log

tests = {}

cur_test = nil
File.readlines(ARGV[0]).each do |line|
  if line =~ /^Running for out\/onnx_real_(.*?)\//
    cur_test = $1
  elsif line =~ /^Elapsed: (\d+\.\d+)/
    tests[cur_test] = $1.to_f
  end
end

tests.each do |name, elapsed|
  puts "#{name} #{elapsed}"
end
