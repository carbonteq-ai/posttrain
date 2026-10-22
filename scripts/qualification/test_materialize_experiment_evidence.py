import hashlib
import io
import tarfile

import pytest
from materialize_experiment_evidence import digest, restore_study


def archive(root, entries, *, manifest_path=None):
    name = 'docs/research/evidence/test-study/result.json'
    path = root / 'evidence.tar.gz'
    with tarfile.open(path, 'w:gz') as bundle:
        for member_name, content in entries:
            info = tarfile.TarInfo(member_name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    payload = entries[0][1]
    return {'name': 'test-study', 'archive': path.name, 'archive_sha256': digest(path),
            'files': [{'path': manifest_path or name, 'bytes': len(payload),
                       'sha256': hashlib.sha256(payload).hexdigest(), 'mode': 0o644}]}


def test_restore_is_lossless_and_idempotent_and_preserves_modified_evidence(tmp_path):
    name = 'docs/research/evidence/test-study/result.json'
    study = archive(tmp_path, [(name, b'{"score":null}\n')])
    assert restore_study(tmp_path, study)['written'] == 0
    assert not (tmp_path / name).exists()
    assert restore_study(tmp_path, study, extract=True)['written'] == 1
    assert (tmp_path / name).read_bytes() == b'{"score":null}\n'
    assert restore_study(tmp_path, study, extract=True)['existing_unchanged'] == 1
    (tmp_path / name).write_text('local research')
    with pytest.raises(ValueError, match='overwrite changed'):
        restore_study(tmp_path, study, extract=True)
    assert (tmp_path / name).read_text() == 'local research'


def test_corrupt_archive_and_wrong_member_digest_are_rejected(tmp_path):
    name = 'docs/research/evidence/test-study/result.json'
    study = archive(tmp_path, [(name, b'data')])
    study['files'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Evidence checksum'):
        restore_study(tmp_path, study, extract=True)
    assert not (tmp_path / name).exists()
    study['archive_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Archive checksum'):
        restore_study(tmp_path, study)


def test_traversal_duplicate_members_and_escaping_symlinks_are_rejected(tmp_path):
    name = 'docs/research/evidence/test-study/result.json'
    study = archive(tmp_path, [('../escape', b'data')], manifest_path='../escape')
    with pytest.raises(ValueError, match='Unsafe evidence path'):
        restore_study(tmp_path, study, extract=True)
    study = archive(tmp_path, [(name, b'data'), (name, b'data')])
    with pytest.raises(ValueError, match='duplicate'):
        restore_study(tmp_path, study)
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    (tmp_path / 'docs').symlink_to(outside, target_is_directory=True)
    study = archive(tmp_path, [(name, b'data')])
    with pytest.raises(ValueError, match='escapes root'):
        restore_study(tmp_path, study, extract=True)
