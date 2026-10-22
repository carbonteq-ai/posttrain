"""Create deterministic, lossless per-study archives; retain all raw originals."""

import argparse
import gzip
import json
import os
import stat
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

from materialize_experiment_evidence import digest, restore_study


def run(root):
    output = root / 'docs/research/experiment-archives'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    groups = defaultdict(list)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for study in previous['studies']:
            for row in study['files']:
                path = root / row['path']
                if digest(path) != row['sha256']:
                    raise ValueError(f'Cannot silently revise historical evidence: {path}')
                groups[study['name']].append(path)
    else:
        names = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z',
                                         'docs/research/evidence/'], cwd=root).decode().split('\0')
        for name in filter(None, names):
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f'Expected regular research artifact: {name}')
            groups[Path(name).relative_to('docs/research/evidence').parts[0]].append(path)
    if not groups:
        raise ValueError('No evidence selected')
    studies = []
    for name, files in sorted(groups.items()):
        archive = output / (name + '.tar.gz')
        temporary = archive.with_suffix('.tmp')
        records = []
        try:
            with temporary.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|') as bundle:
                    for path in sorted(files):
                        relative = path.relative_to(root).as_posix()
                        size = path.stat().st_size
                        sha = digest(path)
                        info = tarfile.TarInfo(relative)
                        info.size = size
                        info.mode = stat.S_IMODE(path.stat().st_mode) & 0o777
                        with path.open('rb') as source:
                            bundle.addfile(info, source)
                        if path.stat().st_size != size or digest(path) != sha:
                            raise ValueError(f'Evidence changed during archival: {relative}')
                        records.append({'path': relative, 'bytes': size, 'sha256': sha, 'mode': info.mode})
            study = {'name': name, 'archive': archive.relative_to(root).as_posix(),
                     'archive_sha256': digest(temporary), 'archive_bytes': temporary.stat().st_size,
                     'original_bytes': sum(r['bytes'] for r in records), 'files': records}
            os.replace(temporary, archive)
            restore_study(root, study)  # Verify every archived byte, not just the source files.
            studies.append(study)
            print(json.dumps({k: v for k, v in study.items() if k != 'files'}), flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {'schema_version': 1, 'purpose': 'Lossless retained experiment evidence; native missing scores remain missing.',
                'raw_files_removed': False, 'studies': studies,
                'files': sum(len(s['files']) for s in studies),
                'original_bytes': sum(s['original_bytes'] for s in studies),
                'archive_bytes': sum(s['archive_bytes'] for s in studies)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'studies'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().root)
