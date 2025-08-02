import time
import random
from motheroffline import PyIVFIndex

# Generate synthetic vectors
def generate_vectors(n: int, dim: int):
    return [(i, [random.random() for _ in range(dim)]) for i in range(n)]

vectors = generate_vectors(10_000, 128)
query = [0.5] * 128

# Create index with 64 centroids
index = PyIVFIndex(64)
index.train(vectors)

# Parallel search
start = time.time()
result_parallel = index.search(query, top_k=10, nprobe=5)
print("Parallel Search:", result_parallel, "Time:", time.time() - start)

# (Optional) Vanilla search for comparison
start = time.time()
result_vanilla = index.search_vanilla(query, top_k=10, nprobe=5)
print("Vanilla Search:", result_vanilla, "Time:", time.time() - start)
