import gzip
import hashlib
import re

import pytest
import yaml

from ingest import download
from ingest.manifest import ManifestError

DATA = b"constant table\n" * 1000
GZ = gzip.compress(b"1\tH2O\t18.0105\t18.0105\n" * 2000)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(download, "RETRY_DELAY_S", 0)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def manifest(tmp_path, *entries) -> str:
    p = tmp_path / "sources.yaml"
    p.write_text(yaml.safe_dump({"sources": list(entries)}), encoding="utf-8")
    return p


def pinned(srv, name="table", body=DATA, layer=1, digest=None):
    srv.files[f"/{name}.txt"] = body
    return {
        "name": name,
        "use": "codata",
        "layer": layer,
        "domains": ["physics"],
        "url": f"{srv.base}/{name}.txt",
        "dest": f"codata/{name}.txt",
        "sha256": digest or sha(body),
        "license": "test",
    }


def rolling(srv, name="mass", body=GZ, upstream_md5=None):
    srv.files[f"/{name}.gz"] = body
    srv.files[f"/{name}.gz.md5"] = f"{upstream_md5 or md5(body)}  {name}.gz\n".encode()
    return {
        "name": name,
        "use": "pubchem-mass",
        "layer": 1,
        "domains": ["chemistry"],
        "url": f"{srv.base}/{name}.gz",
        "dest": f"pubchem/{name}.gz",
        "md5_url": f"{srv.base}/{name}.gz.md5",
        "license": "test",
    }


def log_actions(raw):
    lines = (raw / "download.log").read_text(encoding="utf-8").splitlines()
    for line in lines:
        assert re.fullmatch(r"\[\d\d:\d\d:\d\d\] (start|done|ok|skip|kept|FAIL|BADSUM)\s+\S.*", line), line
    return [line.split()[1] for line in lines]


def test_fresh_download_then_second_run_transfers_nothing(tmp_path, file_server):
    raw = tmp_path / "raw"
    m = manifest(tmp_path, pinned(file_server), rolling(file_server))
    assert download.run(m, raw)
    assert (raw / "codata/table.txt").read_bytes() == DATA
    assert (raw / "pubchem/mass.gz").read_bytes() == GZ
    assert md5(GZ) in (raw / "pubchem/mass.gz.md5").read_text()
    assert (raw / "DONE").exists() and not (raw / "FAILED").exists()
    assert log_actions(raw) == ["start", "done", "ok", "start", "done", "ok"]

    before = len(file_server.requests)
    assert download.run(m, raw)
    assert len(file_server.requests) == before  # sidecar and pinned digest: no network at all
    assert log_actions(raw) == ["skip", "skip"]
    assert (raw / "DONE").exists()


def test_interrupted_download_resumes(tmp_path, file_server):
    raw = tmp_path / "raw"
    m = manifest(tmp_path, pinned(file_server))
    part = raw / "codata" / "table.txt.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(DATA[:5000])
    assert download.run(m, raw)
    assert ("/table.txt", "bytes=5000-") in file_server.requests
    assert (raw / "codata/table.txt").read_bytes() == DATA
    assert not part.exists()


def test_wrong_checksum_fails_and_leaves_no_file(tmp_path, file_server):
    raw = tmp_path / "raw"
    m = manifest(tmp_path, pinned(file_server, digest="f" * 64))
    assert not download.run(m, raw)
    assert not (raw / "codata/table.txt").exists()
    assert not (raw / "codata/table.txt.part").exists()
    assert log_actions(raw) == ["start", "done", "BADSUM", "start", "done", "BADSUM"]
    assert (raw / "FAILED").exists() and not (raw / "DONE").exists()
    assert download.main(["--manifest", str(m), "--raw-dir", str(raw)]) == 1


def test_corrupted_pinned_file_is_replaced(tmp_path, file_server):
    raw = tmp_path / "raw"
    m = manifest(tmp_path, pinned(file_server))
    (raw / "codata").mkdir(parents=True)
    (raw / "codata/table.txt").write_bytes(b"garbage")
    assert download.run(m, raw)
    assert (raw / "codata/table.txt").read_bytes() == DATA


def test_layer_filter(tmp_path, file_server):
    raw = tmp_path / "raw"
    m = manifest(tmp_path, pinned(file_server), pinned(file_server, name="book", layer=2))
    assert download.run(m, raw, layer=1)
    assert file_server.gets("/book.txt") == 0
    assert not (raw / "codata/book.txt").exists()


def test_adopt_file_when_publisher_unchanged(tmp_path, file_server):
    raw = tmp_path / "raw"
    m = manifest(tmp_path, rolling(file_server))
    (raw / "pubchem").mkdir(parents=True)
    (raw / "pubchem/mass.gz").write_bytes(GZ)  # as left by the bootstrap: no sidecar
    assert download.run(m, raw)
    assert file_server.gets("/mass.gz") == 0
    assert md5(GZ) in (raw / "pubchem/mass.gz.md5").read_text()
    assert log_actions(raw) == ["skip"]


def test_adopt_older_snapshot_when_gzip_intact(tmp_path, file_server):
    raw = tmp_path / "raw"
    newer = gzip.compress(b"2\tCH4\t16.0313\t16.0313\n" * 2000)
    m = manifest(tmp_path, rolling(file_server, body=newer))
    (raw / "pubchem").mkdir(parents=True)
    (raw / "pubchem/mass.gz").write_bytes(GZ)
    assert download.run(m, raw)
    assert file_server.gets("/mass.gz") == 0
    assert (raw / "pubchem/mass.gz").read_bytes() == GZ
    assert md5(GZ) in (raw / "pubchem/mass.gz.md5").read_text()
    assert log_actions(raw) == ["kept"]


def test_truncated_gzip_is_downloaded_again(tmp_path, file_server):
    raw = tmp_path / "raw"
    newer = gzip.compress(b"2\tCH4\t16.0313\t16.0313\n" * 2000)
    m = manifest(tmp_path, rolling(file_server, body=newer))
    (raw / "pubchem").mkdir(parents=True)
    (raw / "pubchem/mass.gz").write_bytes(GZ[: len(GZ) // 2])
    assert download.run(m, raw)
    assert (raw / "pubchem/mass.gz").read_bytes() == newer
    assert log_actions(raw) == ["start", "done", "ok"]


def test_invalid_manifest_makes_no_request(tmp_path, file_server):
    bad = pinned(file_server)
    bad["md5"] = "0" * 32  # two checksum sources
    with pytest.raises(ManifestError):
        download.run(manifest(tmp_path, bad), tmp_path / "raw")
    assert file_server.requests == []


def test_missing_url_fails_entry_but_not_run(tmp_path, file_server):
    raw = tmp_path / "raw"
    gone = pinned(file_server, name="gone")
    del file_server.files["/gone.txt"]
    m = manifest(tmp_path, gone, pinned(file_server))
    assert not download.run(m, raw)
    assert (raw / "codata/table.txt").exists()
    assert log_actions(raw) == ["start", "FAIL", "start", "done", "ok"]
