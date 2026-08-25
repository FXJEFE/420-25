# -*- coding: utf-8 -*-
"""
FXJEFE persistent I/O helpers — UTF-8 + robust JSON/CSV.

Use everywhere live data / config is read or written so Windows, MT5 and
Python stay on one encoding contract:

  * JSON  : UTF-8 (no BOM), ensure_ascii=False, trailing newline
  * CSV   : UTF-8 (no BOM), comma separator, LF or platform newline
  * JSON text from HTTP/WebRequest: strip BOM, repair trailing junk
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

PathLike = Union[str, Path]

# Preferred encodings for read (order matters)
_JSON_READ_ENCS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_CSV_READ_ENCS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def enable_utf8() -> None:
    """Force UTF-8 stdio / env for Windows Python processes."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
    for name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


enable_utf8()


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser()


def strip_bom(text: str) -> str:
    if text and text[0] == "\ufeff":
        return text[1:]
    return text


def read_text(path: PathLike, encodings: Sequence[str] = _JSON_READ_ENCS) -> str:
    """Read text file trying encodings; always returns a str (may use replace)."""
    p = _as_path(path)
    raw = p.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def write_text_utf8(path: PathLike, text: str, newline: Optional[str] = "\n") -> Path:
    """Write UTF-8 without BOM. Creates parent dirs."""
    p = _as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = text if text.endswith("\n") or text == "" else text + "\n"
    # Explicit no-BOM UTF-8
    with p.open("w", encoding="utf-8", newline=newline or "") as f:
        f.write(data if data.endswith("\n") else data + "\n")
    return p


# ── JSON ──────────────────────────────────────────────────────────────────────

def parse_json_text(text: str) -> Any:
    """
    Parse JSON from string robustly (config files, AI responses, WebRequest body).
    - Strips BOM / whitespace
    - Accepts first JSON value if trailing garbage after a valid object/array
    """
    s = strip_bom(text or "").strip()
    if not s:
        raise ValueError("empty JSON text")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Extract first complete object or array
        start_obj = s.find("{")
        start_arr = s.find("[")
        if start_obj < 0 and start_arr < 0:
            raise
        if start_obj < 0:
            start = start_arr
            open_c, close_c = "[", "]"
        elif start_arr < 0:
            start = start_obj
            open_c, close_c = "{", "}"
        else:
            if start_arr < start_obj:
                start = start_arr
                open_c, close_c = "[", "]"
            else:
                start = start_obj
                open_c, close_c = "{", "}"
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    return json.loads(s[start : i + 1])
        raise


def load_json(path: PathLike, default: Any = None) -> Any:
    """Load JSON file with UTF-8-first decoding. default= if missing (else raises)."""
    p = _as_path(path)
    if not p.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(str(p))
    return parse_json_text(read_text(p))


def save_json(path: PathLike, obj: Any, indent: int = 2) -> Path:
    """Persist JSON as UTF-8 (no BOM), ensure_ascii=False, stable newlines."""
    p = _as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False)
    if not text.endswith("\n"):
        text += "\n"
    # write bytes without BOM
    p.write_bytes(text.encode("utf-8"))
    return p


def validate_json_file(path: PathLike) -> Dict[str, Any]:
    """Return {ok, path, error?, keys?} after parse attempt."""
    p = _as_path(path)
    out: Dict[str, Any] = {"ok": False, "path": str(p)}
    try:
        data = load_json(p)
        out["ok"] = True
        if isinstance(data, dict):
            out["keys"] = list(data.keys())[:40]
            out["type"] = "object"
        elif isinstance(data, list):
            out["type"] = "array"
            out["len"] = len(data)
        else:
            out["type"] = type(data).__name__
    except Exception as e:
        out["error"] = str(e)
    return out


# ── CSV ───────────────────────────────────────────────────────────────────────

def read_csv_rows(path: PathLike) -> List[List[str]]:
    """Read CSV as list of rows (strings), encoding-robust."""
    text = read_text(path, _CSV_READ_ENCS)
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.reader(io.StringIO(text))
    return [row for row in reader]


def write_csv_rows(path: PathLike, rows: Iterable[Sequence[Any]], header: Optional[Sequence[Any]] = None) -> Path:
    """Write CSV UTF-8 no BOM."""
    p = _as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if header is not None:
        w.writerow(list(header))
    for row in rows:
        w.writerow(list(row))
    p.write_bytes(buf.getvalue().encode("utf-8"))
    return p


def read_csv_df(path: PathLike, **kwargs):
    """pandas read_csv with encoding fallbacks. Requires pandas."""
    import pandas as pd

    p = _as_path(path)
    last_err = None
    for enc in _CSV_READ_ENCS:
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False, **kwargs)
        except Exception as e:
            last_err = e
            continue
    # last resort: decode replace then parse
    text = read_text(p, _CSV_READ_ENCS)
    try:
        return pd.read_csv(io.StringIO(text), low_memory=False, **kwargs)
    except Exception:
        if last_err:
            raise last_err
        raise


def write_csv_df(path: PathLike, df, index: bool = False) -> Path:
    """pandas to_csv UTF-8 no BOM."""
    p = _as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, encoding="utf-8", lineterminator="\n")
    return p


def normalize_csv_utf8(path: PathLike, backup: bool = True) -> Dict[str, Any]:
    """
    Re-write a live CSV as clean UTF-8 (no BOM). Preserves rows.
    Returns status dict.
    """
    p = _as_path(path)
    out: Dict[str, Any] = {"ok": False, "path": str(p)}
    if not p.is_file():
        out["error"] = "missing"
        return out
    try:
        rows = read_csv_rows(p)
        if backup:
            bak = p.with_suffix(p.suffix + ".bak_utf8")
            if not bak.exists():
                bak.write_bytes(p.read_bytes())
                out["backup"] = str(bak)
        write_csv_rows(p, rows)
        # Verify re-parse
        _ = read_csv_rows(p)
        head = p.read_bytes()[:3]
        out["ok"] = True
        out["rows"] = len(rows)
        out["bom"] = head == b"\xef\xbb\xbf"
        out["utf8"] = True
    except Exception as e:
        out["error"] = str(e)
    return out


def live_feature_csv_paths(config: Optional[dict] = None) -> List[Path]:
    """Known live FXJEFE_Features.csv locations (project + MT5)."""
    paths: List[Path] = []
    docs = Path(r"C:\Users\locallarry\Documents")
    proj = docs / "FXJEFE_Project"
    name = "FXJEFE_Features.csv"
    if config is None:
        for cand in (docs / "config.json", proj / "config.json"):
            if cand.is_file():
                try:
                    config = load_json(cand)
                    break
                except Exception:
                    pass
    config = config or {}

    def add(p: Optional[str]) -> None:
        if not p:
            return
        pp = Path(p)
        if pp.is_dir():
            paths.append(pp / name)
        else:
            paths.append(pp)

    add(str(proj / "data" / name))
    add(config.get("mt5_files_path"))
    add(config.get("mt5_common_path"))
    add(config.get("data_path"))
    add(config.get("data_output_path"))
    # Common MT5 terminals from config
    for t in config.get("mt5_terminals") or []:
        if isinstance(t, dict):
            add(t.get("files_path"))
    # Always include known terminal IDs
    term_root = Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal"
    if term_root.is_dir():
        for d in term_root.iterdir():
            if d.is_dir():
                paths.append(d / "MQL5" / "Files" / name)
        paths.append(term_root / "Common" / "Files" / name)

    # unique existing + non-existing targets we care about
    seen = set()
    out: List[Path] = []
    for p in paths:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def ensure_project_io(config_paths: Optional[Sequence[PathLike]] = None) -> Dict[str, Any]:
    """
    Validate/rewrite project configs as UTF-8 JSON and normalize live feature CSVs.
    Safe to run repeatedly.
    """
    docs = Path(r"C:\Users\locallarry\Documents")
    proj = docs / "FXJEFE_Project"
    if config_paths is None:
        config_paths = [
            docs / "config.json",
            proj / "config.json",
            docs / "FXJEFE_Project" / "config" / "config.yaml",  # skip non-json later
        ]
    report: Dict[str, Any] = {"json": [], "csv": [], "ok": True}

    for cp in config_paths:
        p = _as_path(cp)
        if p.suffix.lower() not in (".json",):
            continue
        if not p.is_file():
            report["json"].append({"path": str(p), "ok": False, "error": "missing"})
            continue
        try:
            data = load_json(p)
            # re-persist clean UTF-8
            save_json(p, data)
            v = validate_json_file(p)
            report["json"].append(v)
            if not v.get("ok"):
                report["ok"] = False
        except Exception as e:
            report["ok"] = False
            report["json"].append({"path": str(p), "ok": False, "error": str(e)})

    cfg = None
    for item in report["json"]:
        if item.get("ok"):
            try:
                cfg = load_json(item["path"])
                break
            except Exception:
                pass

    for csv_path in live_feature_csv_paths(cfg):
        if not csv_path.is_file():
            continue
        r = normalize_csv_utf8(csv_path, backup=True)
        report["csv"].append(r)
        if not r.get("ok"):
            report["ok"] = False

    return report


if __name__ == "__main__":
    enable_utf8()
    rep = ensure_project_io()
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    sys.exit(0 if rep.get("ok") else 1)
