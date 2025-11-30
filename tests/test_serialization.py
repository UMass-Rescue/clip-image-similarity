from dataclasses import dataclass
from pathlib import Path

import json

from clip_image_similarity.serialization import _to_jsonable, save_config, save_json


def test_to_jsonable_converts_paths_and_nested():
    data = {
        "path": Path("/tmp/file"),
        "list": [Path("/tmp/a"), {"nested": Path("/tmp/b")}],
        "value": 3,
    }
    result = _to_jsonable(data)
    assert result["path"] == "/tmp/file"
    assert result["list"][0] == "/tmp/a"
    assert result["list"][1]["nested"] == "/tmp/b"


@dataclass
class DummyConfig:
    a: int
    b: Path


def test_save_json_writes_file(tmp_path, capsys):
    out = tmp_path / "data" / "obj.json"
    save_json({"a": 1}, out)
    assert json.loads(out.read_text()) == {"a": 1}
    assert "Saved JSON" in capsys.readouterr().out


def test_save_config_writes_config_json(tmp_path, monkeypatch):
    @dataclass
    class Config:
        a: int
        b: Path

    cfg = Config(a=1, b=tmp_path)
    # Monkeypatch log to avoid printing
    monkeypatch.setattr("clip_image_similarity.serialization.log", lambda msg: None)
    save_config(cfg, tmp_path)
    cfg_path = tmp_path / "config.json"
    contents = json.loads(cfg_path.read_text())
    assert contents == {"a": 1, "b": str(tmp_path)}
