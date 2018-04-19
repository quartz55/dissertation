import contextlib
import numbers
import subprocess as sp
import io
from fractions import Fraction
from subprocess import Popen, DEVNULL, PIPE
import json
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any, Union, List
from collections import OrderedDict
import copy
import shlex

FFmpegArg = Union[str, Tuple[str, Any]]
FFmpegIO = Dict[str, Union[None, str, List[str]]]


class FFmpeg:
    def __init__(self, executable='ffmpeg',
                 inputs: FFmpegIO = None,
                 outputs: FFmpegIO = None,
                 global_args: List[FFmpegArg] = None):
        self._executable: str = executable
        self._inputs: FFmpegIO = inputs if inputs is not None else {}
        self._outputs: FFmpegIO = outputs if outputs is not None else {}
        global_args = global_args if global_args is not None else []
        self._global_args = [arg for arg in global_args if not isinstance(arg, tuple)]
        self._global_kargs = {karg[0]: karg[1] for karg in global_args if isinstance(karg, tuple)}

    def run(self, stdin=None, stdout=None, stderr=None, auto_pipes=True) -> Popen:
        def check_pipes(inputs_or_outputs):
            return next((True for i in inputs_or_outputs.keys() if 'pipe:' in i or i == '-'), False)

        cmd = self._build_command()
        if auto_pipes:
            if stdin is None and check_pipes(self._inputs):
                stdin = PIPE
            if stdout is None and check_pipes(self._outputs):
                stdout = PIPE

        return Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr)

    @property
    def inputs(self) -> Dict[str, str]:
        return self._inputs

    @property
    def outputs(self) -> Dict[str, str]:
        return self._outputs

    @property
    def cmd(self) -> str:
        return ' '.join(self._build_command())

    def _build_command(self) -> List[str]:
        global_args = [f'-{arg}'
                       for arg in self._global_args]
        for arg, val in self._global_kargs.items():
            global_args.extend([f'-{arg}', str(val)])

        input_args = []
        for name, args in self._inputs.items():
            if args is not None:
                if isinstance(args, str):
                    input_args.extend(shlex.split(args or ''))
                elif isinstance(args, list):
                    input_args.extend(args)
                else:
                    raise ValueError(f"Invalid input arguments type '{type(args)}' (must be a str or list)")
            input_args.extend(['-i', name])

        output_args = []
        for name, args in self._outputs.items():
            if args is not None:
                if isinstance(args, str):
                    output_args.extend(shlex.split(args or ''))
                elif isinstance(args, list):
                    output_args.extend(args)
                else:
                    raise ValueError(f"Invalid output arguments type '{type(args)}' (must be a str or list)")
            output_args.append(name)

        cmd = [self._executable]
        cmd.extend(global_args)
        cmd.extend(input_args)
        cmd.extend(output_args)
        return cmd


class FFprobe(FFmpeg):
    def __init__(self, executable='ffprobe',
                 inputs: FFmpegIO = None,
                 global_args: List[FFmpegArg] = None):
        super().__init__(executable, inputs, None, global_args)

    def json(self) -> Dict[str, Any]:
        if 'print_format' in self._global_kargs:
            del self._global_kargs['print_format']
        self._global_kargs['of'] = 'json'
        with self.run(stdout=PIPE, stderr=PIPE) as proc:
            out, _ = proc.communicate()
            return json.loads(out)


class Frame(np.ndarray):
    def __new__(cls, input_array, metadata=None):
        if not isinstance(input_array, np.ndarray):
            raise ValueError('Frame expects a numpy array.')
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('Frame expects metadata to be a dict.')
        try:
            obj = input_array.view(cls)
        except AttributeError:
            return input_array

        metadata = metadata if metadata is not None else {}
        obj._metadata: Dict[str, Any] = copy.deepcopy(metadata)
        return obj

    def __array_finalize__(self, obj):
        if isinstance(obj, Frame):
            self._metadata = copy.deepcopy(obj.meta)
        else:
            self._metadata = {}

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def meta(self) -> Dict[str, Any]:
        return self._metadata

    def to_image(self, mode: str = None) -> Image.Image:
        return Image.fromarray(self, mode=mode)


class Reader(contextlib.AbstractContextManager):
    def __init__(self, path: str):
        probe = FFprobe(global_args=[('v', 'quiet'), 'show_format', 'show_streams'],
                        inputs={path: ''})
        probe_info = probe.json()
        self.streams = probe_info['streams']
        self.format = probe_info['format']
        self.metadata = self.streams[0]
        self.metadata['fps'] = self.metadata['avg_frame_rate']
        self.metadata['frames'] = self.metadata['nb_frames']

        self.path: str = path
        self._process: Popen = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.metadata['width'], self.metadata['height']

    @property
    def stream(self) -> Optional[io.BufferedReader]:
        return self._process.stdout if self._process is not None else None

    def open(self):
        ffmpeg = FFmpeg(global_args=[('v', 'quiet')],
                        inputs={self.path: ''},
                        outputs={'pipe:1': '-f rawvideo -c:v rawvideo -pix_fmt rgb24'})
        self._process = ffmpeg.run(stderr=DEVNULL)

    def read(self) -> Optional[Frame]:
        if self._process is None:
            raise ValueError("Reader must be open to read")
        if self._process.poll() is not None:
            return None

        w, h = self.shape
        raw_frame = self.stream.read(w * h * 3)
        if not raw_frame:
            return None

        frame = np.fromstring(raw_frame, dtype=np.uint8).reshape((h, w, 3))
        frame = Frame(frame, metadata=self.metadata)
        return frame

    def close(self):
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._process.__exit__(None, None, None)
            self._process = None
            return

        self._process.__exit__(None, None, None)
        self._process.kill()
        self._process = None

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.metadata['frames'])

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class Writer(contextlib.AbstractContextManager):
    def __init__(self, path: str,
                 fps=Fraction(24000, 1001), codec='libx264',
                 bitrate=None, pixel_format='yuv420p',
                 shape=None, quality=5):
        self.path: str = path
        self._fps: Fraction = None
        self.fps = fps
        self.codec: str = codec
        self.bitrate: int = bitrate
        self.quality: int = quality
        self.pixel_format: str = pixel_format
        self._shape: Optional[Tuple[int, int]] = shape
        self._process: Optional[Popen] = None
        self._input_meta: Dict[str, Any] = {
            'pix_fmt': None,
            'size': None,
            'depth': None
        }

    @property
    def fps(self) -> Fraction:
        return self._fps

    @fps.setter
    def fps(self, fps: Union[numbers.Number, str, Fraction]):
        self._fps = fps if isinstance(fps, Fraction) else Fraction(fps)

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        return self._shape

    def _open(self):
        w, h = self._input_meta['size']
        size_str = f'{w}x{h}'
        out_cmd = ['-an',
                   '-c:v', self.codec,
                   '-pix_fmt', self.pixel_format,
                   '-f', 'mp4']
        if self.bitrate is not None:
            out_cmd.extend(['-b:v', str(self.bitrate)])
        elif self.quality is not None:
            quality = 1 - self.quality / 10.0
            if self.codec == 'libx264':
                quality = int(quality * 51)
                out_cmd.extend(['-crf', str(quality)])
            else:
                quality = int(quality * 30) + 1
                out_cmd.extend(['-qscale:v', str(quality)])

        ffmpeg = FFmpeg(global_args=['y', ('v', 'warning')],
                        inputs={'pipe:0': ['-f', 'rawvideo',
                                           '-c:v', 'rawvideo',
                                           '-s', size_str,
                                           '-pix_fmt', self._input_meta['pix_fmt'],
                                           '-r', str(self.fps)]},
                        outputs={self.path: out_cmd})
        self._process = ffmpeg.run(stdout=None)

    def write(self, frame: np.ndarray):
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")

        h, w = frame.shape[:2]
        d = 1 if frame.ndim == 2 else frame.shape[2]

        if self._process is None:
            if self._input_meta['size'] is None:
                formats = {1: 'gray', 2: 'gray8a', 3: 'rgb24', 4: 'rgba'}
                self._input_meta['pix_fmt'] = formats.get(d, None)
                if self._input_meta['pix_fmt'] is None:
                    raise ValueError(f"Invalid number of channels in frame {d} (must be between 1 and 4)")
                self._input_meta['size'] = (w, h)
                self._input_meta['depth'] = d
                self._open()
            else:
                raise RuntimeError("Trying to write to closed Writer")

        if (w, h) != self._input_meta['size']:
            raise ValueError('All frames should have the same size')
        if d != self._input_meta['depth']:
            raise ValueError('All frames should have the same number of channels')

        self._process.stdin.write(frame.tostring())

    def __exit__(self, exc_type, exc_value, traceback):
        if self._process is None:
            return
        if self._process.poll() is not None:
            return
        if self._process.stdin:
            self._process.stdin.close()
        self._process.wait()
        self._process = None

