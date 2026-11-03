from pathlib import Path
import pytest
import run_pretrained


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(run_pretrained.os, 'environ', dict(run_pretrained.os.environ))


def test_research_defaults_to_separate_database(monkeypatch):
    monkeypatch.delenv('MEDISCAN_DB', raising=False)
    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    db = run_pretrained.configure_environment()
    assert db == Path('output/research/mediscan.db').resolve()


def test_research_refuses_operational_database(monkeypatch):
    monkeypatch.setenv('MEDISCAN_DB', str(Path('data/mediscan.db').resolve()))
    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    with pytest.raises(SystemExit, match='cannot use'):
        run_pretrained.configure_environment()


def test_research_honors_disposable_database(monkeypatch, tmp_path):
    monkeypatch.setenv('MEDISCAN_DB', str(tmp_path / 'test.db'))
    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    assert run_pretrained.configure_environment() == tmp_path / 'test.db'
