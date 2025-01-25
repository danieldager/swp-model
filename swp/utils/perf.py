import time
from collections import defaultdict
from functools import wraps


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        name = func.__name__
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        total = end - start
        buf = 25 - len(name)
        print(f'{name}: {" "*buf} {total:.2f} seconds')
        return result

    return timeit_wrapper


class Timer:
    def __init__(self):
        self.starts = {}
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def start(self, operation):
        self.starts[operation] = time.time()

    def stop(self, operation):
        elapsed = time.time() - self.starts[operation]
        self.times[operation] += elapsed
        self.counts[operation] += 1

    def summary(self):
        print("\nTiming Summary:")
        print("-" * 60)
        print(f"{'Operation':<30} {'Total (s)':<15}")
        print("-" * 60)
        for operation in self.times:
            print(f"{operation:<30} {self.times[operation]:>13.3f}s")
