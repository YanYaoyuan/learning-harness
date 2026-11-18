from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import relocate_large_files as migration


def test_verified_atomic_relocation_preserves_open_readers(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, "WORK", tmp_path)
    source = tmp_path / "old/weights.bin"
    source.parent.mkdir()
    source.write_bytes(b"frozen weights" * 1024)
    target = tmp_path / "disk/weights.bin"
    with source.open("rb") as reader:
        result = migration.relocate(source, target)
        assert source.is_symlink()
        assert source.read_bytes() == target.read_bytes() == reader.read()
    assert result["sha256"] == migration.sha(target)


def test_destination_collision_keeps_original(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, "WORK", tmp_path)
    source, target = tmp_path / "source.bin", tmp_path / "target.bin"
    source.write_bytes(b"original")
    target.write_bytes(b"different")
    with pytest.raises(ValueError, match="collision"):
        migration.relocate(source, target)
    assert not source.is_symlink()
    assert source.read_bytes() == b"original"
    assert target.read_bytes() == b"different"


def test_preexisting_verified_copy_can_finish_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, "WORK", tmp_path)
    source, target = tmp_path / "source.bin", tmp_path / "target.bin"
    source.write_bytes(b"same")
    target.write_bytes(b"same")
    migration.relocate(source, target)
    assert source.is_symlink()
    assert source.read_bytes() == b"same"
