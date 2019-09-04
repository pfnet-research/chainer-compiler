import time


def run_benchmark(fn, iterations):
    elapsed_times = []
    if iterations > 1:
        num_iterations = iterations - 1
        for t in range(num_iterations):
            start = time.time()
            fn()
            elapsed_times.append(time.time() - start)
        print('Elapsed: %.3f msec' % (sum(elapsed_times) * 1000 / num_iterations))
    return elapsed_times


def load_test_data(data_dir, input_names, output_names):
    inout_values = []
    for kind, names in [('input', input_names), ('output', output_names)]:
        names = list(names)
        values = []
        for pb in sorted(glob.glob(os.path.join(data_dir, '%s_*.pb' % kind))):
            with open(pb, 'rb') as f:
                tensor = onnx.TensorProto()
                tensor.ParseFromString(f.read())
            if tensor.name in names:
                name = tensor.name
                names.remove(name)
            else:
                name = names.pop(0)
            values.append((name, onnx.numpy_helper.to_array(tensor)))
        inout_values.append(values)
    return tuple(inout_values)
