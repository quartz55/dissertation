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
    def __init__(self, name: str, alias: str = None):
        super().__init__(connection_pool=pool)
        self._key = ""
        self._name = ""
        self.name = name
        self._empty = True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self._key = f"{self._name}:{BENCH_ID}"

    @property
    def key(self):
        return self._key

    def _setup(self):
        if self._empty:
            self._empty = False
            self.sadd(f"benches:id:{BENCH_ID}", self.key)

    def measurement(self, group: str, value: float):
        if self._empty:
            self._setup()
        key_name = self.key + ":" + group
        return self.rpush(key_name, value)

    def measurements(self):
        if self._empty:
            self._setup()
        return _Measurements(self.key, self)

