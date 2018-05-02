import numpy as np
from functools import partial
import numbers
from typing import Any, Tuple, Dict, List, Union, TypeVar, Type

FIELD_NAMES = ['x', 'y', 'z', 'w']


class VecMeta(type):
    def __new__(mcs, typename: str, bases: Tuple[type, ...], ns: Dict[str, Any]):
        if ns.get('_base', False):
            return super().__new__(mcs, typename, bases, ns)

        defaults = {
            'size': next((getattr(base, '_size') for base in bases if hasattr(base, '_size')), None),
            'dtype': next((getattr(base, '_dtype') for base in bases if hasattr(base, '_dtype')), None),
        }

        attrs = {**defaults, **ns}
        if 'size' in ns:
            del ns['size']
        if 'dtype' in ns:
            del ns['dtype']

        if 'size' not in attrs or not isinstance(attrs['size'], int) or attrs['size'] < 1:
            raise TypeError(f"Must specify a Vec size >= 1 (was {attrs['size']})")
        if 'dtype' not in attrs:
            raise TypeError("Must specify Vec data type")

        size = ns['_size'] = attrs['size']
        ns['_dtype'] = attrs['dtype']

        def pget(key: int, self):
            return self._data[key]

        def pset(key: int, self, value):
            try:
                value = self._dtype(value)
            except ValueError:
                raise TypeError(f"Provided value can't be converted: {self._dtype}({value})")
            self._data[key] = value

        for i, name in enumerate(FIELD_NAMES[slice(0, size)]):
            ns[name] = property(partial(pget, i), partial(pset, i))

        ns['__slots__'] = []

        return super().__new__(mcs, typename, bases, ns)


VecType = TypeVar('VecType', bound='Vec')
VecArgs = Union[np.ndarray, VecType, List[numbers.Number]]
Scalar = numbers.Number


class Vec(metaclass=VecMeta):
    _base: bool = True
    _size: int
    _dtype: Type = np.float64

    __slots__ = ['_data']

    def __init__(self, *args: VecArgs, **kwargs: _dtype) -> None:
        self._data = np.zeros(self._size, self._dtype)
        if len(args) > 0:
            data = args[0]
            if isinstance(data, np.ndarray):
                assert len(data) == self._size
                self._data = np.array(data, dtype=self._dtype)
            elif isinstance(data, Vec):
                assert data._size <= self._size
                sl1 = slice(0, self._size)
                self._data[sl1] = data._data[sl1]
            else:
                assert len(args) <= self._size
                for i, v in enumerate(args):
                    self._data[i] = v

        indexes = {name: i for i, name in enumerate(FIELD_NAMES)}
        for k, v in kwargs.items():
            if k not in FIELD_NAMES:
                raise ValueError(f"Key '{k}' is invalid")
            if indexes[k] >= self._size:
                raise ValueError(f"Key '{k}' is out of range ({self._size})")

            try:
                v = self._dtype(v)
            except ValueError:
                raise TypeError(f"Provided value can't be converted: {self._dtype}({v})")
            setattr(self, k, self._dtype(v))

    @property
    def length(self) -> float:
        return float(np.around(np.sqrt(self._data.dot(self._data)), 15))

    def normalized(self, length: Scalar = 1) -> VecType:
        mag = self.length
        res = self.__class__() if mag == 0 else self.__class__(self._data / mag)
        return res * length

    def dot(self, other: VecType) -> float:
        assert isinstance(other, Vec)
        return float(np.dot(self._data, other._data))

    @classmethod
    def max(cls, v1: VecType, v2: VecType) -> VecType:
        return cls(np.amax(np.vstack((v1._data, v2._data)), axis=0))

    @classmethod
    def min(cls, v1: VecType, v2: VecType) -> VecType:
        return cls(np.amin(np.vstack((v1._data, v2._data)), axis=0))

    def __add__(self, scalar_or_vec: Union[Scalar, VecType]) -> VecType:
        if isinstance(scalar_or_vec, Vec):
            return self.__class__(self._data + scalar_or_vec._data)
        return self.__class__(self._data + scalar_or_vec)

    def __sub__(self, scalar_or_vec: Union[Scalar, VecType]) -> VecType:
        if isinstance(scalar_or_vec, Vec):
            return self.__class__(self._data - scalar_or_vec._data)
        return self.__class__(self._data - scalar_or_vec)

    def __neg__(self) -> VecType:
        return self.__class__(-self._data)

    def __mul__(self, scalar_or_vec: Union[Scalar, VecType]) -> VecType:
        if isinstance(scalar_or_vec, Vec):
            return self.__class__(self._data * scalar_or_vec._data)
        return self.__class__(self._data * scalar_or_vec)

    def __truediv__(self, scalar_or_vec: Union[Scalar, VecType]) -> VecType:
        if isinstance(scalar_or_vec, Vec):
            return self.__class__(self._data / scalar_or_vec._data)
        return self.__class__(self._data / scalar_or_vec)

    def __getitem__(self, item: int) -> _dtype:
        if not isinstance(item, int) or not (0 <= item < self._size):
            raise ValueError(f"Index out of range {item}/{self._size}")
        return self._data[item]

    def __setitem__(self, key: int, value: Scalar):
        assert isinstance(key, int)
        try:
            value = self._dtype(value)
        except ValueError:
            raise TypeError(f"Provided value can't be converted: {self._dtype}({value})")
        self._data[key] = value

    def __iter__(self):
        for v in self._data:
            yield v

    def __str__(self) -> str:
        aux = f"{self.__class__.__name__}({self._size}, {self._dtype}) "
        if self._size <= len(FIELD_NAMES):
            aux += "("
            for k in FIELD_NAMES[slice(0, self._size)]:
                aux += f"{k}={getattr(self, k)} "
            aux = aux[:-1] + ")"

        else:
            aux += str(self._data)
        return aux

    def __repr__(self) -> str:
        return str(self)


class Vec2(Vec):
    size = 2


class Vec3(Vec):
    size = 3


class Vec4(Vec):
    size = 4


def vector(size: int, dtype: Type = np.float64) -> Type[VecType]:
    s, dt = size, dtype

    class VecN(Vec):
        size = s
        dtype = dt

    return VecN


def make_vector(size: int, dtype: Type = np.float64, *args: VecArgs, **kwargs: Dict[str, Any]) -> VecType:
    return vector(size, dtype)(*args, **kwargs)
