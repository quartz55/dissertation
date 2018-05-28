import asyncio
import logging
import os
import tempfile as tmp
import time
import urllib.parse
from typing import Tuple, List, Dict

import aiofiles
import aiohttp
import psutil
from aiostream import stream
from tqdm import tqdm

from cimc.core import log

WEIGHTS = {
    # 'https://pjreddie.com/media/files/yolov2.weights': 'yolov2.weights',
    'https://pjreddie.com/media/files/yolov3.weights': 'yolov3.weights',
}

TEST_FILES = {
    'http://ipv4.download.thinkbroadband.com/20MB.zip': '20MB.zip',
    'http://ipv4.download.thinkbroadband.com/50MB.zip': '50MB.zip',
    'http://ipv4.download.thinkbroadband.com/100MB.zip': '100MB.zip',
}

logger = logging.getLogger(__name__)
logger.addHandler(log.TqdmLoggingHandler())
logger.setLevel(logging.DEBUG)


class UrlNoByteRangesError(Exception):
    __slots__ = []


class SegmentChunk:
    __slots__ = ['data', 'index', 'range']

    def __init__(self, data, index: int, range: Tuple[int, int]):
        self.data = data
        self.index = index
        self.range = range

    @property
    def size(self):
        return len(self.data)

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.size}bytes, " \
               f"index={self.index}, " \
               f"range={self.range[0]}-{self.range[1]})"

    def __str__(self):
        return self.__repr__()


class Segment:
    __slots__ = [
        'chunk_size', '_temp_file', '_connection', 'done',
        'index', 'range', '_timings'
    ]

    def __init__(self,
                 connection: aiohttp.ClientResponse,
                 chunk_size=1024 * 1024,
                 index: int = None,
                 range: Tuple[int, int] = None):
        self.chunk_size = chunk_size
        self._temp_file = tmp.NamedTemporaryFile(
            prefix=f"{tmp.gettempprefix()}.s{index}",
            suffix='tmp',
            dir=os.getcwd())
        self._connection: aiohttp.ClientResponse = connection
        self.done = False
        self.index = index
        self.range = range
        self._timings = dict()
        logger.debug(f"Created temporary segment file for {self}")

    @property
    def temp_file(self):
        return self._temp_file

    async def close(self):
        self._temp_file.close()
        await self._connection.release()

    def __aiter__(self):
        self._timings['start'] = time.time()
        return self

    async def __anext__(self):
        chunk = await self._connection.content.read(self.chunk_size)
        if not chunk:
            self._timings['end'] = time.time()
            self.done = True
            logger.debug(
                f"{self} downloaded in {round(self._timings['end']-self._timings['start'])}s"
            )
            raise StopAsyncIteration
        self._temp_file.write(chunk)
        return SegmentChunk(chunk, self.index, self.range)

    def __repr__(self):
        a, b = self.range
        return f"{self.__class__.__name__}(index={self.index}, " \
               f"range={a}-{b}({b-a} bytes), " \
               f"tmp_file={self._temp_file.name}))"

    def __str__(self):
        return self.__repr__()


class Download:
    __slots__ = [
        'filename', 'filepath', 'uri', 'headers', '_file', '_session',
        '_own_session', '_response', '_prepared'
    ]

    def __init__(self,
                 uri: str,
                 filename: str = None,
                 directory: str = None,
                 session: aiohttp.ClientSession = None):
        if filename is None:
            filename = urllib.parse.urlparse(uri).path
            filename = os.path.basename(filename)
        if directory is None:
            directory = os.getcwd()
        self.filename = filename
        self.filepath = os.path.normpath(os.path.join(os.getcwd(), directory, filename))
        self.uri = uri
        self.headers: Dict[str, str] = None
        self._file = None
        self._session = session
        self._own_session = session is None
        self._response: aiohttp.ClientResponse = None
        self._prepared = False

    @property
    def size(self):
        assert self.headers is not None
        return int(self.headers['content-length'])

    async def __aiter__(self):
        await self.prepare()
        return self

    async def __anext__(self):
        chunk = await self._response.content.read(1024 * 1024)
        if not chunk:
            raise StopAsyncIteration
        await self._file.write(chunk)
        return chunk

    async def close(self):
        if self._file is not None and not self._file.closed:
            await self._file.close()
        if self._response is not None and not self._response.closed:
            await self._response.release()
        if (self._session is not None and self._own_session
                and not self._session.closed):
            await self._session.close()

    async def prepare(self):
        if not self._prepared:
            self._prepared = True
            self._check_session()
            self._response = await self._session.get(self.uri)
            self._mkdir_p()
            self._file = await aiofiles.open(self.filepath, 'wb')
            self.headers = self._response.headers
        return self

    def _check_session(self):
        if self._session is None and self._own_session:
            self._session = aiohttp.ClientSession()

    def _mkdir_p(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def __await__(self):
        return self.prepare().__await__()

    async def __aenter__(self):
        await self.prepare()
        return self

    async def __aexit__(self, *args):
        await self.close()


class AccelDownload(Download):
    __slots__ = ['num_segments', '_segments']

    def __init__(self, *args, segments: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_segments = segments
        self._segments: List[Segment] = []

    async def __aiter__(self):
        async def gen():
            chunks_stream = stream.merge(
                *[stream.iterate(segm) for segm in self._segments])
            async with chunks_stream.stream() as chunks:
                async for chunk in chunks:
                    yield chunk

        return gen()

    async def __anext__(self):
        raise NotImplementedError

    async def prepare(self):
        if not self._prepared:
            self._prepared = True
            self._check_session()
            self.headers = (await self._session.head(self.uri)).headers
            if 'accept-ranges' not in self.headers or 'bytes' not in self.headers['accept-ranges']:
                raise UrlNoByteRangesError
            segment_size = self.size // self.num_segments
            ranges: List[Tuple[int, int]] = []
            for i in range(self.num_segments):
                start = i * segment_size
                end = self.size if i == self.num_segments - 1 else (
                                                                           (i + 1) * segment_size) - 1
                ranges.append((start, end))
            logger.debug(
                f"Preparing {self.num_segments} requests for '{self.uri}'")
            for i, r in enumerate(ranges):
                byte_range = f"bytes={r[0]}-{r[1]}"
                conn = await self._session.get(
                    self.uri, headers={'range': byte_range})
                self._segments.append(Segment(conn, index=i, range=r))
            basename = os.path.basename(self.uri)
            logger.debug(
                f"'{basename}' ready for download (size={self.size} bytes, "
                f"num_segments={self.num_segments}, "
                f"segment_size={segment_size}, "
                f"segments={ranges})")
        return self

    async def close(self):
        all_done = next((False for s in self._segments if not s.done), True)
        if all_done:
            self._mkdir_p()
            async with aiofiles.open(self.filepath, 'wb') as file:
                logger.info(
                    f"Merging downloaded segments({self.num_segments}) into final file '{self.filepath}'"
                )
                for segment in self._segments:
                    logger.debug(
                        f"Merging segment {segment.index} of '{self.uri}'")
                    segment.temp_file.seek(0)
                    while True:
                        buf = segment.temp_file.read(1024 * 1024)
                        if not buf:
                            break
                        await file.write(buf)
                    await segment.close()
        else:
            logger.warning(
                f"Not all segments were downloaded successfully. Not saving file to disk"
            )
            for s in self._segments:
                await s.close()
        await super().close()


def _download_tqdm(down: Download, **kwargs):
    defaults = dict(total=down.size,
                    desc=f"Downloading '{down.filename}",
                    dynamic_ncols=True,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024)
    opts = {**defaults, **kwargs}
    return tqdm(**opts)


async def download(uri: str,
                   filename: str = None,
                   dir: str = None,
                   segments: int = None,
                   overwrite: bool = False,
                   session: aiohttp.ClientSession = None):
    if segments is None or segments > 1:
        if segments is None:
            segments = psutil.cpu_count(logical=True)
        down = AccelDownload(uri,
                             filename=filename,
                             directory=dir,
                             segments=segments,
                             session=session)
        if not overwrite and os.access(down.filepath, os.W_OK):
            logger.info(f"File '{down.filepath}' already downloaded")
            return uri, down.filepath
        elif overwrite:
            logger.warning(f"File '{down.filepath}' exists, overwritting...")
        async with down:
            with _download_tqdm(down,
                                postfix=dict(segments=down.num_segments)) as bar:
                async for segment_chunk in down:
                    bar.update(len(segment_chunk))
    else:
        down = Download(uri,
                        filename=filename,
                        directory=dir,
                        session=session)
        if not overwrite and os.access(down.filepath, os.W_OK):
            logger.info(f"File '{down.filepath}' already downloaded")
            return uri, down.filepath
        elif overwrite:
            logger.warning(f"File '{down.filepath}' exists, overwritting...")
        async with down:
            with _download_tqdm(down) as bar:
                async for chunk in down:
                    bar.update(len(chunk))


def download_sync(*args, **kwargs):
    (asyncio.get_event_loop()
     .run_until_complete(download(*args, **kwargs)))


async def download_queue(downloads: List[Tuple[str, str]],
                         directory: str = None,
                         parallel: bool = False,
                         segments: int = None):
    if segments is None:
        segments = psutil.cpu_count(logical=True)
    if directory is None:
        directory = os.getcwd()
    async with aiohttp.ClientSession() as sess:
        queue = [stream.just(download(uri,
                                      filename,
                                      directory,
                                      segments,
                                      session=sess))
                 for uri, filename in downloads]
        if parallel:
            async with stream.merge(*queue).stream() as queue:
                async for uri, outpath in queue:
                    logger.info(
                        f"[PARALLEL] Successfully downloaded '{uri}' ({outpath})"
                    )
        else:
            async with stream.chain(*queue).stream() as queue:
                async for uri, outpath in queue:
                    logger.info(f"Successfully downloaded '{uri}' ({outpath})")


def download_queue_sync(*args, **kwargs):
    (asyncio.get_event_loop()
     .run_until_complete(download_queue_sync(*args, **kwargs)))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'directory',
        nargs='?',
        type=str,
        default='.',
        help="directory to download the weights to")
    parser.add_argument(
        '-p',
        '--parallel',
        action='store_true',
        help="download files in parallel")
    parser.add_argument(
        '-s',
        '--segments',
        type=int,
        help="split each file into segments and download concurrently")
    args = parser.parse_args()
    (asyncio
     .get_event_loop()
     .run_until_complete(download_queue(WEIGHTS.items(),
                                        None,
                                        args.parallel,
                                        args.segments)))


if __name__ == '__main__':
    main()
