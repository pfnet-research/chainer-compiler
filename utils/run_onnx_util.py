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
