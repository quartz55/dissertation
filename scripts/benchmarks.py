import numpy as np

import cimc.utils.bench as metrics

conn = metrics._conn


def _bench_results(bench_type: str):
    keys = map(lambda k: k.decode("utf-8"), conn.keys(f"{bench_type}:*"))
    sub_types = {}
    for key in keys:
        sub_type = key.split(":")[-1]
        if sub_types.get(sub_type) is None:
            sub_types[sub_type] = []
        sub_types[sub_type] += map(float, conn.lrange(key, 0, -1))

    print(bench_type)
    for sub in sub_types:
        print(f"  {sub}")

    return {"bench": bench_type,
            "subs": sub_types.keys(),
            "timings": sub_types}


def _results_mean(results):
    print(results["bench"])
    for n, t in results["timings"].items():
        mean = np.mean(t)
        print(f"  {n}: {mean*1e3:.3f} ms")
