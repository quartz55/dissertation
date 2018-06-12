import os.path
from typing import Dict, Any, Tuple, List, Optional
from cimc.scene.classification import SceneClassification
from cimc.tracker import TrackedBoundingBox
import imageio
from imageio.core import Format


class VideoMetaMixin:
    _metadata: Dict[str, Any] = {"fps": 0.0, "size": (0, 0)}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, meta: Dict[str, Any]):
        assert isinstance(meta, dict)
        assert {"fps", "size"}.issubset(meta.keys())
        self._metadata = meta

    @property
    def fps(self) -> float:
        return self.metadata["fps"]

    @property
    def length(self) -> int:
        return self.metadata["nframes"]

    @property
    def duration(self) -> float:
        return self.metadata["duration"] or self.length / self.fps

    @property
    def size(self) -> Tuple[int, int]:
        return self.metadata["size"]

    @property
    def shape(self) -> Tuple[int, int, int]:
        w, h = self.size
        return h, w, 3


class Segment(VideoMetaMixin):
    __slots__ = ["id", "start", "end", "scene", "objects"]

    def __init__(self, id: int, start: int, metadata: Dict[str, Any] = None):
        self.id = id
        self.start = start
        self.end: Optional[int] = None
        self.scene: Optional[SceneClassification] = None
        self.objects: List[Dict[int, List[TrackedBoundingBox]]] = []
        if metadata is not None:
            self.metadata = metadata

    def __len__(self):
        if self.end is None:
            raise ValueError("Segment has no 'end' frame")
        return self.end - self.start

    def append_objects(self, objects: Dict[int, List[TrackedBoundingBox]]):
        self.objects.append(objects)


class VideoClassification(VideoMetaMixin):
    __slots__ = ["filename", "name", "segments"]

    def __init__(
        self,
        file_uri: str,
        segments: List[Segment] = None,
        name: str = None,
        metadata: Dict[str, Any] = None,
        reader: Format.Reader = None,
    ):
        self.segments = [] if segments is None else segments

        self.filename = os.path.basename(file_uri)

        if name is None:
            name = os.path.splitext(self.filename)[0]
        self.name = name

        if metadata is None:
            if reader is None and os.path.isfile(file_uri):
                try:
                    # type: Format.Reader
                    with imageio.get_reader(file_uri) as reader:
                        metadata = reader.get_meta_data()
                except (ValueError, RuntimeError):
                    pass
            elif reader is not None:
                metadata = reader.get_meta_data()
        assert metadata is not None, "Couldn't get metadata"
        self.metadata = metadata

    def append_segment(self, segment: Segment):
        segment.metadata = self.metadata
        self.segments.append(segment)

    def __iter__(self):
        return iter(self.segments)
