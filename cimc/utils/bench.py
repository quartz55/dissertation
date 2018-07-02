import redis

pool = redis.ConnectionPool()

_conn = redis.StrictRedis(connection_pool=pool)
_counter_key = "global:bench:counter"
BENCH_ID = int(_conn.incr(_counter_key))


def new_batch():
    global BENCH_ID
    BENCH_ID = int(_conn.incr(_counter_key))
    return BENCH_ID


class _Measurements:
    def __init__(self, key: str, conn: redis.StrictRedis):
        self.__key = key
        self.__pipe = conn.pipeline()

    def add(self, group: str, value: float):
        key_name = self.__key + ":" + group
        self.__pipe.rpush(key_name, value)
        return self

    def done(self):
        return self.__pipe.execute()


class Bench(redis.StrictRedis):
    def __init__(self, name: str, alias: str=None):
        super().__init__(connection_pool=pool)
        self.name = name
        self._key = f"{name}:{BENCH_ID}"
        self.sadd(f"benches:type:{name}", self._key)
        self.sadd(f"benches:id:{BENCH_ID}", self._key)

    def measurement(self, group: str, value: float):
        key_name = self._key + ":" + group
        return self.rpush(key_name, value)

    def measurements(self):
        return _Measurements(self._key, self)
