import time
import random
from motheroffline import PyIVFIndex

print("\n== Synthetic (10K x 128D) Benchmark ==")

def generate_vectors(n: int, dim: int):
    return [(i, [random.uniform(0, 1) for _ in range(dim)]) for i in range(n)]

vectors_syn = generate_vectors(10_000, 128)
index = PyIVFIndex(64)
index.train(vectors_syn)

query = [random.uniform(0, 1) for _ in range(128)]

# Parallel = search()
start = time.time()
results_parallel = index.search_parallel(query, 10, 5)
time_parallel = time.time() - start

# Vanilla = search_vanilla()
start = time.time()
results_vanilla = index.search_vanilla(query, 10, 5)
time_vanilla = time.time() - start

# Output
print("\nResults (top-3 shown):")
for p, v in zip(results_parallel[:3], results_vanilla[:3]):
    print(f"Parallel: {p},  Vanilla: {v}")

print("\nTiming:")
print(f"Parallel Search: {time_parallel:.6f} sec")
print(f"Vanilla Search : {time_vanilla:.6f} sec")
