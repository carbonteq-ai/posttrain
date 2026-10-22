"""Verify or restore lossless research evidence archives without overwriting work."""

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError(f'Unsafe evidence path: {name}')
    destination = root / name
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Evidence path escapes root: {name}')
    return destination


def restore_study(root, study, *, extract=False):
    archive = contained(root, study['archive'])
    if digest(archive) != study['archive_sha256']:
        raise ValueError(f'Archive checksum differs: {study["name"]}')
    expected = {row['path']: row for row in study['files']}
    if len(expected) != len(study['files']):
        raise ValueError('Duplicate manifest paths')
    for name in expected:
        contained(root, name)
    seen, written, existing = set(), 0, 0
    with tarfile.open(archive, 'r|gz') as bundle:
        for member in bundle:
            name = member.name
            if name not in expected or name in seen or not member.isfile():
                raise ValueError(f'Unexpected, duplicate or nonregular member: {name}')
            seen.add(name)
            row = expected[name]
            if member.size != row['bytes']:
                raise ValueError(f'Evidence size differs: {name}')
            destination = contained(root, name)
            skip = False
            if extract and destination.exists():
                if destination.is_symlink() or not destination.is_file() or digest(destination) != row['sha256']:
                    raise ValueError(f'Refusing to overwrite changed evidence: {name}')
                skip = True
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f'Missing archive member stream: {name}')
            temporary = None
            try:
                if extract and not skip:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    contained(root, name)
                    temporary = tempfile.NamedTemporaryFile(dir=destination.parent, delete=False)
                sha = hashlib.sha256()
                for chunk in iter(lambda stream=stream: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
                    if temporary:
                        temporary.write(chunk)
                if sha.hexdigest() != row['sha256']:
                    raise ValueError(f'Evidence checksum differs: {name}')
                if temporary:
                    temporary.close()
                    os.chmod(temporary.name, row.get('mode', 0o644) & 0o777)
                    # Atomic no-clobber install. Never replace a file that
                    # appeared while this archive member was being verified.
                    os.link(temporary.name, destination)
                    written += 1
                elif skip:
                    existing += 1
            finally:
                if temporary:
                    temporary.close()
                    Path(temporary.name).unlink(missing_ok=True)
    if seen != set(expected):
        raise ValueError(f'Archive is missing {len(set(expected) - seen)} manifest members')
    return {'study': study['name'], 'verified_files': len(seen), 'written': written, 'existing_unchanged': existing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--manifest', default='docs/research/experiment-archives/manifest.json')
    parser.add_argument('--study', action='append', help='Restore only these study directory names; repeatable.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true')
    action.add_argument('--verify', action='store_true')
    action.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(contained(args.root, args.manifest).read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported evidence manifest schema')
    studies = manifest['studies']
    if args.study:
        if set(args.study) - {s['name'] for s in studies}:
            raise ValueError('Unknown study selection')
        studies = [s for s in studies if s['name'] in args.study]
    for study in studies:
        result = ({k: study[k] for k in ('name', 'archive', 'original_bytes', 'archive_bytes')}
                  if args.list else restore_study(args.root, study, extract=args.extract))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
