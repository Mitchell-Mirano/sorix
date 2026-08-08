"""hotspot_profile.py – identifies the slowest functions in the backward pass."""
from __future__ import annotations
import cProfile, pstats, io
import numpy as np
from sorix.tensor import Tensor

def scenario_backward():
    a = Tensor(np.random.randn(512, 512).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(512, 512).astype(np.float32), requires_grad=True)
    # typical MLP-like chain: matmul → sigmoid → sum → backward
    for _ in range(20):
        out = (a @ b).sigmoid().sum()
        out.backward()
        a.grad = None; b.grad = None

pr = cProfile.Profile()
pr.enable()
scenario_backward()
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(25)
print(s.getvalue())

with open("tests/benchmark/hotspot_report.txt", "w") as f:
    f.write(s.getvalue())
print("Saved to tests/benchmark/hotspot_report.txt")
