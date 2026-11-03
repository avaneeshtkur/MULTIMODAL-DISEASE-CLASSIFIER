"""Download a public pinned artifact with bounded range requests and SHA256.

Never executes downloaded code. Existing completed targets are only accepted
when their hash matches. Partial chunks are retained for interrupted downloads.
"""
import argparse
import concurrent.futures
import hashlib
from pathlib import Path
import shutil
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--size', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prefix', type=Path)
    args = p.parse_args()
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    if args.out.exists():
        if sha(args.out) != args.sha256:
            raise ValueError('Existing target hash mismatch; refusing overwrite')
        return
    chunks = args.out.with_suffix(args.out.suffix + '.chunks')
    chunks.mkdir(parents=True, exist_ok=True)
    start = args.prefix.stat().st_size if args.prefix else 0
    if not 0 <= start <= args.size:
        raise ValueError('Invalid partial prefix size')
    ranges = [(n, min(n + 8 * 1024 * 1024, args.size) - 1)
              for n in range(start, args.size, 8 * 1024 * 1024)]
    def download(bounds):
        lo, hi = bounds
        path = chunks / f'{lo}-{hi}'
        if path.exists() and path.stat().st_size == hi - lo + 1:
            return path
        request = urllib.request.Request(args.url, headers={'Range': f'bytes={lo}-{hi}'})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {lo}-{hi}/{args.size}':
                raise ValueError('Server did not honor requested byte range')
            with path.open('wb') as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
        if path.stat().st_size != hi - lo + 1:
            raise ValueError('Incomplete range')
        print(f'Completed bytes {lo}-{hi}', flush=True)
        return path
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pieces = list(pool.map(download, ranges))
    temporary = args.out.with_suffix(args.out.suffix + '.assembling')
    with temporary.open('wb') as out:
        for path in ([args.prefix] if args.prefix else []) + pieces:
            with path.open('rb') as source:
                shutil.copyfileobj(source, out, length=1024 * 1024)
    if temporary.stat().st_size != args.size or sha(temporary) != args.sha256:
        raise ValueError('Final artifact integrity check failed; not activated')
    temporary.rename(args.out)
    print('Verified:', args.out, flush=True)


if __name__ == '__main__':
    main()
