#!/bin/env fades

import asyncio
import os
import urllib.parse
import shutil
import tempfile as tmp
from typing import Tuple, List

import aiofiles  # fades
import aiohttp  # fades
from aiostream import stream  # fades
from tqdm import tqdm  # fades

WEIGHTS = {
    'yolo2.weights': 'https://pjreddie.com/media/files/yolov2.weights',
    'yolo3.weights': 'https://pjreddie.com/media/files/yolov3.weights'
}


class UrlNoByteRangesError(Exception):
    pass


class Segment:
    def __init__(self, connection: aiohttp.ClientResponse, chunk_size=1024):
        self.chunk_size = chunk_size
        self._temp_file = tmp.TemporaryFile()
        self._connection: aiohttp.ClientResponse = connection
        self.done = False

    @property
    def temp_file(self):
        return self._temp_file

    def close(self):
        self._temp_file.close()
        self._connection.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self._connection.content.read(self.chunk_size)
        if not chunk:
            self.done = True
            raise StopAsyncIteration
        self._temp_file.write(chunk)
        return chunk


class Accelerator:
    def __init__(self, session: aiohttp.ClientSession, request: aiohttp.ClientResponse,
                 uri: str, path: str,
                 segments: List[Segment]):
        self._request = request
        self._session = session
        self._segments = segments
        self.path = path
        self.uri = uri

    @property
    def size(self):
        return int(self._request.headers.get('content-length', 0))

    @classmethod
    async def accelerate(cls, uri: str, path: str, segments: int = 4):
        session = aiohttp.ClientSession()
        req = await session.head(uri)
        if 'accept-ranges' not in req.headers or 'bytes' not in req.headers['accept-ranges']:
            raise UrlNoByteRangesError
        size = int(req.headers.get('content-length', 0))
        segment_size = size // segments
        ranges: List[Tuple[int, int]] = []
        for i in range(segments):
            start = i * segment_size
            end = size if i == segments - 1 else ((i + 1) * segment_size) - 1
            ranges.append((start, end))
        segments_list = []
        for r in ranges:
            conn = await session.get(uri, headers={'range': f"bytes={r[0]}-{r[1]}"})
            segments_list.append(Segment(conn))
        return cls(session, req, uri, path, segments_list)

    async def packets(self):
        chunks_stream = stream.merge(*[stream.iterate(segm) for segm in self._segments])
        async with chunks_stream.stream() as chunks:
            async for chunk in chunks:
                yield chunk

    def close(self):
        with open(self.path, 'wb') as final:
            for segment in self._segments:
                shutil.copyfileobj(segment.temp_file, final)
                segment.close()
        self._request.close()
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()


async def download_weights(uri: str, filename: str = None, dir: str = './',
                           segments: int = None) -> str:
    if filename is None:
        filename = urllib.parse.urlparse(uri).path
        filename = os.path.basename(filename)
    filepath = os.path.normpath(os.path.join(os.getcwd(), dir, filename))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if segments is not None:
        assert segments > 1
        accel = await Accelerator.accelerate(uri, filepath, segments)
        with accel:
            with tqdm(total=accel.size, desc=f"Downloading weights '{filename}'",
                      unit='B', unit_scale=True, unit_divisor=1024) as bar:
                async for packet in accel.packets():
                    bar.update(len(packet))
    else:
        async def to_stream(response, chunk_size=1024):
            while True:
                chunk = await response.content.read(chunk_size)
                if not chunk:
                    return
                yield chunk
        async with aiohttp.ClientSession() as session:
            async with session.get(uri) as res:
                size = res.headers['content-length']
                async with aiofiles.open(filepath, 'wb') as fd:
                    with tqdm(total=int(size), desc=f"Downloading weights '{filename}'",
                              unit='B', unit_scale=True, unit_divisor=1024) as bar:
                        async for chunk in to_stream(res):
                            fd.write(chunk)
                            bar.update(len(chunk))
                        bar.set_description(f"Done '{filename}'")
    return filepath


async def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory', nargs='?', type=str,
                        default='.',
                        help="directory to download the weights to")
    parser.add_argument('-p', '--parallel', action='store_true',
                        help="download files in parallel")
    parser.add_argument('-s', '--segments', type=int,
                        help="split each file into segments and download concurrently")
    args = parser.parse_args()
    if args.parallel:
        async def write_success(future):
            name = await future
            tqdm.write(f"Successfully downloaded '{name}'")

        futures = [write_success(download_weights(uri, filename, args.directory, args.segments))
                   for filename, uri in WEIGHTS.items()]
        await asyncio.gather(*futures)
    else:
        for filename, uri in WEIGHTS.items():
            res = await download_weights(uri, filename, args.directory, args.segments)
            tqdm.write(f"Successfully downloaded '{res}'")


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
