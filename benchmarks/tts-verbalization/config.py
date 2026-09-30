"""YAML config -> adapter instances.

Every TTS entry names its `manifest` and its `audio_root` SEPARATELY and
explicitly. Nothing is inferred from folder names, because the real layout has
a trap: `polynorm_tts/manifest.csv` describes the flux run while
`polynorm_tts/af_heart/` holds Kokoro's audio. Pairing by folder would label
Kokoro's numbers as flux.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.tts.http import _expand_env  # reuse; do not write a second one
from schema import Item


def load_config(path: Path | str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a mapping")
    return cfg


def build_items(cfg: dict) -> list[Item]:
    ds = cfg.get("dataset", {})
    kind = ds.get("type", "polynorm")
    if kind == "polynorm":
        from data.loaders.polynorm import DEFAULT_PATH, load
    elif kind == "synthetic":
        from data.loaders.synthetic import DEFAULT_PATH, load
    else:
        raise ValueError(f"unknown dataset type {kind!r} (expected polynorm or synthetic)")
    items = load(ds.get("path", DEFAULT_PATH))
    # An item with no text is not a test, and sending it to a provider is
    # actively harmful: polynorm-282 has empty written AND spoken fields, and
    # every model rejects it -- some by erroring ("Text input cannot be
    # empty"), at least one by HANGING. Combined with --retry-errors that
    # produced a livelock: retry the empty item, hang, watchdog restarts,
    # retry the empty item again. Drop it at the source.
    usable = [i for i in items if i.written.strip()]
    dropped = len(items) - len(usable)
    if dropped:
        import sys
        blank = ", ".join(i.id for i in items if not i.written.strip())
        print(f"dataset: dropped {dropped} item(s) with empty text ({blank})",
              file=sys.stderr)
    return usable


def dataset_path(cfg: dict) -> "Path":
    """The file build_items() will actually read.

    run.py hashes this into the manifest. It used to import
    polynorm.DEFAULT_PATH directly, which meant a synthetic run
    recorded the checksum of a corpus it never opened -- the one
    field whose whole job is provenance.
    """
    ds = cfg.get("dataset", {})
    kind = ds.get("type", "polynorm")
    if kind == "polynorm":
        from data.loaders.polynorm import DEFAULT_PATH
    elif kind == "synthetic":
        from data.loaders.synthetic import DEFAULT_PATH
    else:
        raise ValueError(f"unknown dataset type {kind!r}")
    return Path(ds.get("path", DEFAULT_PATH))

def build_tts(cfg: dict, items: list[Item], include_partial: bool = False) -> list:
    from adapters.tts.folder import FolderTTS
    from adapters.tts.http import HttpTTS
    from adapters.tts.manifest import ManifestTTS

    out = []
    for entry in cfg.get("tts", []):
        # Skip BEFORE expanding: a disabled example block must not require its
        # ${VARS} to be set. Expanding first made api_example.yaml unloadable
        # unless you had credentials for every provider listed in it.
        if entry.get("enabled") is False and not include_partial:
            continue
        entry = _expand_env(entry)
        kind = entry.get("type", "manifest")
        if kind == "manifest":
            root = Path(entry["audio_root"]).expanduser()
            man = Path(entry["manifest"]).expanduser()
            for p, what in ((root, "audio_root"), (man, "manifest")):
                if not p.exists():
                    raise FileNotFoundError(
                        f"tts {entry['id']!r}: {what} not found: {p}")
            a = ManifestTTS(entry["id"], man, root,
                            revision=str(entry.get("revision", "unknown")))
            a.verify_against(items)          # fails loudly on a text mismatch
        elif kind == "folder":
            a = FolderTTS(entry["folder"], id=entry["id"],
                          revision=str(entry.get("revision", "n/a")))
        elif kind == "http":
            a = HttpTTS(entry)
        elif kind == "replicate":
            from adapters.tts.replicate import ReplicateTTS
            a = ReplicateTTS(entry)
        else:
            raise ValueError(f"unknown tts type {kind!r}")
        out.append(a)
    return out


def build_asr(cfg: dict) -> list:
    out = []
    for entry in cfg.get("asr", []):
        # Same rule as build_tts, and for the same reason: skip BEFORE the
        # adapter is constructed, so a disabled provider block does not demand
        # its ${VARS} be set. Without this, a config listing two API
        # recognizers is unloadable until you hold both keys.
        if entry.get("enabled") is False:
            continue
        kind = entry.get("type", "wav2vec2")
        if kind == "wav2vec2":
            from adapters.asr.wav2vec2 import DEFAULT_MODEL, Wav2Vec2ASR
            out.append(Wav2Vec2ASR(entry.get("model", DEFAULT_MODEL),
                                   device=entry.get("device", "cpu")))
        elif kind == "http":
            from adapters.asr.http import HttpASR
            out.append(HttpASR(entry))
        elif kind == "whisper":
            from adapters.asr.whisper import DEFAULT_MODEL as W_DEFAULT
            from adapters.asr.whisper import WhisperASR
            out.append(WhisperASR(entry.get("model", W_DEFAULT),
                                  device=entry.get("device", "cpu")))
        else:
            raise ValueError(f"unknown asr type {kind!r}")
    return out


def build_adapters(cfg: dict, include_partial: bool = False):
    items = build_items(cfg)
    return items, build_tts(cfg, items, include_partial), build_asr(cfg)
