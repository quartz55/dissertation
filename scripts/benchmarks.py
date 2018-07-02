import numpy as np

import cimc.utils.bench as metrics

conn = metrics._conn


def bench_results(bench_type: str):
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


def bench_results_id(bench_id: int):
    types = conn.smembers(f"benches:id:{bench_id}")
    res = {"id": bench_id,
           "subs": {}}
    for t in types:
        sub_name = "".join(t.decode("utf-8").split(":")[:-1])
        sub_types = {}
        for k in conn.keys(t + b":*"):
            sub_type = k.decode("utf-8").split(":")[-1]
            if sub_types.get(sub_type) is None:
                sub_types[sub_type] = []
            sub_types[sub_type] += map(float, conn.lrange(k, 0, -1))
        res["subs"][sub_name] = sub_types

    print(bench_id)
    for s, ss in res["subs"].items():
        print("  " + s)
        for ss_n, ss_t in ss.items():
            print(f"    {ss_n}: {np.mean(ss_t)*1e3:.3f} ms")


def results_mean(results):
    print(results["bench"])
    for n, t in results["timings"].items():
        mean = np.mean(t)
        print(f"  {n}: {mean*1e3:.3f} ms")
