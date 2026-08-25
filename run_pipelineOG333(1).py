
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE Trading Pipeline OG333 — locallarry edition
Runs starter scripts from Documents, config-driven, AI server first.
"""
from __future__ import annotations
import os as _os_utf8, sys as _sys_utf8
_os_utf8.environ.setdefault('PYTHONUTF8','1')
_os_utf8.environ.setdefault('PYTHONIOENCODING','utf-8')
for _s in (getattr(_sys_utf8,'stdout',None), getattr(_sys_utf8,'stderr',None)):
    try:
        if _s is not None and hasattr(_s,'reconfigure'):
            _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import argparse
import json
import logging
import os
import subprocess
import sys
import time

try:
    import requests
except ImportError:  # optional until health check
    requests = None

DEFAULT_CONFIG = os.path.join(
    os.environ.get("USERPROFILE", r"C:\Users\locallarry"),
    "Documents",
    "config.json",
)

pipeline_scripts = [
    "fill_all_csvs.py",
    "create_structure.py",
    "mt5_data_sync.py",
    "fix_csv.py",
    "fix_csv_encoding.py",
    "convert_encoding.py",
    "generate_new_csv.py",
    "process_trades.py",
    "merge_datasets.py",
    "generate_labels.py",
    "feature_engineering.py",
    "clean_training_data.py",
    "generate_training_data.py",
    "Load_and_Process.py",
    "validate_data.py",
    "train_models.py",
    "train_model_zoo.py",
    "ensemble_predictions.py",
    "generate_signals_with_xgboost.py",
    "get_lstm_prediction.py",
    "fxjefe_xgboost_api.py",
    "check_integrity.py",
    "check_labels.py",
    "log_summary.py",
    "parse_log_to_csv.py",
    "risk_management.py",
    "signal_processor.py",
    "mt5_signal_script.py",
    "update_database.py",
    "update_scripts.py",
    "test_encoding.py",
    "test_regex.py",
    "test_server.py",
    "waitress server.py",
    "logging_utils.py",
    "verify_m15_align.py",
]

optional_scripts = [
    "adjust_headers.py",
    "analyze_outcomes.py",
]


def setup_logging(log_path: str) -> None:
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "pipelineOG333.log")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_server_health(config: dict) -> bool:
    if requests is None:
        logging.warning("requests not installed; skipping health check body parse")
        return False
    url = config.get("ai_server_url", "http://127.0.0.1:8080").rstrip("/") + "/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            try:
                body = response.json()
                if body.get("status") in ("running", "ok", "healthy", True):
                    logging.info("AI server is running")
                    return True
            except Exception:
                logging.info("AI server responded 200")
                return True
        logging.error("AI server not healthy: %s", response.status_code)
        return False
    except Exception as e:
        logging.error("Server health check failed: %s", e)
        return False


def resolve_venv_python(config: dict) -> str:
    """Prefer project venv so every child script uses the same packages."""
    project = config.get("project_root") or os.path.join(
        os.environ.get("USERPROFILE", r"C:\Users\locallarry"),
        "Documents",
        "FXJEFE_Project",
    )
    candidates = [
        config.get("python_executable"),
        os.path.join(project, "venv", "Scripts", "python.exe"),
        os.path.join(project, ".venv", "Scripts", "python.exe"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return sys.executable


def venv_env(config: dict, python_exe: str) -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    scripts_path = config.get("scripts_path") or ""
    env["PYTHONPATH"] = os.pathsep.join(
        [scripts_path, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    venv_dir = os.path.dirname(os.path.dirname(python_exe))
    if os.path.isdir(os.path.join(venv_dir, "Scripts")) or os.path.isdir(
        os.path.join(venv_dir, "bin")
    ):
        env["VIRTUAL_ENV"] = venv_dir
        bindir = os.path.join(venv_dir, "Scripts")
        if not os.path.isdir(bindir):
            bindir = os.path.join(venv_dir, "bin")
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["FXJEFE_PROTECT_OG_MODELS"] = "1"
    return env


def start_ai_server(config: dict) -> bool:
    scripts_path = config["scripts_path"]
    script_path = os.path.join(scripts_path, config.get("ai_server_script", "ai_server.py"))
    if not os.path.exists(script_path):
        logging.error("AI server script missing: %s", script_path)
        return False
    try:
        creation = 0
        if sys.platform == "win32":
            creation = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
        py = resolve_venv_python(config)
        env = venv_env(config, py)
        subprocess.Popen(
            [py, "-X", "utf8", script_path],
            cwd=scripts_path,
            creationflags=creation,
            env=env,
        )
        logging.info("Started AI server: %s", script_path)
        time.sleep(6)
        return True
    except Exception as e:
        logging.error("Failed to start AI server: %s", e)
        return False


def _banner(title: str, char: str = "=", width: int = 72) -> None:
    line = char * width
    # Print directly so output is visible even if logging is filtered
    print(f"\n{line}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{line}", flush=True)
    logging.info(title)


def _print_stream_block(label: str, text: str, script: str) -> None:
    """Print full sub-script stream to console + log (no truncation)."""
    if text is None:
        text = ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        print(f"  [{script}] {label}: (empty)", flush=True)
        logging.info("[%s] %s: (empty)", script, label)
        return
    print(f"  ---- {script} {label} begin ----", flush=True)
    # Mirror every line so each sub-script is readable in the runner console
    for line in text.split("\n"):
        print(f"  | {line}", flush=True)
    print(f"  ---- {script} {label} end ----", flush=True)
    logging.info("[%s] %s:\n%s", script, label, text)


def run_script(
    script: str,
    config: dict,
    index: int = 0,
    total: int = 0,
    show_output: bool = True,
) -> str:
    """Run one pipeline script; stream full stdout/stderr. Return 'ok'|'skip'|'fail'."""
    scripts_path = config["scripts_path"]
    script_path = os.path.join(scripts_path, script)
    log_dir = config.get("log_path") or os.path.join(scripts_path, "logs")
    script_log_dir = os.path.join(log_dir, "pipeline_scripts")
    os.makedirs(script_log_dir, exist_ok=True)

    if not os.path.exists(script_path):
        _banner(f"[{index}/{total}] FAIL (missing): {script}", char="!")
        logging.error("Script %s not found — not skipped silently: %s", script, script_path)
        return "fail"

    progress = f"[{index}/{total}]" if total else ""
    _banner(f"{progress} RUN: {script}")
    print(f"  path: {script_path}", flush=True)
    print(f"  cwd:  {scripts_path}", flush=True)

    t0 = time.time()
    try:
        py = config.get("_python_exe") or resolve_venv_python(config)
        env = venv_env(config, py)

        timeout = int(config.get("script_timeout_sec", 1800))
        # Live-stream child output so user sees each line as it happens
        proc = subprocess.Popen(
            [py, "-X", "utf8", "-u", script_path],
            cwd=scripts_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )

        # Read both streams without deadlock via communicate with timeout
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            elapsed = time.time() - t0
            if show_output:
                _print_stream_block("STDOUT", out or "", script)
                _print_stream_block("STDERR", err or "", script)
            _banner(f"{progress} TIMEOUT ({elapsed:.1f}s): {script}", char="!")
            logging.error("Timeout executing %s after %.1fs", script, elapsed)
            return "fail"

        elapsed = time.time() - t0
        out = out or ""
        err = err or ""

        # Per-script log file (full output)
        safe_name = script.replace(" ", "_").replace("/", "_")
        per_log = os.path.join(script_log_dir, f"{safe_name}.log")
        try:
            with open(per_log, "w", encoding="utf-8", errors="replace") as lf:
                lf.write(f"# script: {script}\n")
                lf.write(f"# path: {script_path}\n")
                lf.write(f"# exit: {proc.returncode}\n")
                lf.write(f"# elapsed_sec: {elapsed:.3f}\n\n")
                lf.write("===== STDOUT =====\n")
                lf.write(out)
                if not out.endswith("\n"):
                    lf.write("\n")
                lf.write("\n===== STDERR =====\n")
                lf.write(err)
                if err and not err.endswith("\n"):
                    lf.write("\n")
        except OSError as e:
            logging.warning("Could not write per-script log %s: %s", per_log, e)

        if show_output:
            _print_stream_block("STDOUT", out, script)
            _print_stream_block("STDERR", err, script)

        print(f"  exit_code: {proc.returncode}", flush=True)
        print(f"  elapsed:   {elapsed:.2f}s", flush=True)
        print(f"  log_file:  {per_log}", flush=True)

        if proc.returncode == 0 and not (out or "").strip() and not (err or "").strip():
            _banner(f"{progress} FAIL (silent empty output): {script}", char="!")
            logging.error("%s produced no stdout/stderr — refusing silent skip", script)
            return "fail"

        if proc.returncode != 0:
            _banner(
                f"{progress} FAIL (code={proc.returncode}, {elapsed:.1f}s): {script}",
                char="!",
            )
            logging.error(
                "%s failed (code %s) in %.2fs — see %s",
                script,
                proc.returncode,
                elapsed,
                per_log,
            )
            return "fail"

        _banner(f"{progress} OK ({elapsed:.1f}s): {script}", char="-")
        logging.info("Successfully executed %s in %.2fs", script, elapsed)
        return "ok"

    except Exception as e:
        elapsed = time.time() - t0
        _banner(f"{progress} ERROR ({elapsed:.1f}s): {script} — {e}", char="!")
        logging.error("Error executing %s: %s", script, e)
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FXJEFE Trading Pipeline OG333")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.json")
    parser.add_argument("--retry", type=int, default=2, help="Retries for failed scripts")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        default=True,
        help="Always include optional scripts (default True — do not skip)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue pipeline if a script fails (default True for full sweep)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on first hard failure after retries",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Do not require AI server health",
    )
    parser.add_argument(
        "--quiet-scripts",
        action="store_true",
        help="Do not echo each sub-script stdout/stderr (still writes per-script logs)",
    )
    args = parser.parse_args()
    continue_on_error = not args.strict
    show_output = not args.quiet_scripts

    if not os.path.exists(args.config):
        print(f"Error: config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    for key in ("scripts_path", "log_path", "models_path"):
        if not config.get(key):
            print(f"Error: config missing required key: {key}")
            sys.exit(1)
        os.makedirs(config[key], exist_ok=True)

    setup_logging(config["log_path"])
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    py = resolve_venv_python(config)
    config["_python_exe"] = py
    logging.info("Pipeline OG333 starting")
    logging.info("config=%s scripts_path=%s python=%s", args.config, config["scripts_path"], py)
    config["ai_server_url"] = config.get("ai_server_url") or "http://127.0.0.1:8080"

    # Snapshot OG models — refuse to leave them mutated
    try:
        from fxjefe_paths import snapshot_model_files, verify_og_models_untouched
        og_snap = snapshot_model_files(config)
        logging.info("OG model snapshot: %s protected files", len(og_snap))
    except Exception as e:
        logging.warning("Could not snapshot OG models: %s", e)
        og_snap = {}

    if not args.skip_server:
        if not check_server_health(config):
            logging.info("AI server not running, attempting to start it")
            if not start_ai_server(config) or not check_server_health(config):
                logging.warning(
                    "AI server failed to start or health check failed; continuing with --continue-on-error"
                )
                if args.strict:
                    logging.error("AI server required in strict mode; aborting")
                    sys.exit(1)

    scripts_to_run = list(pipeline_scripts) + list(optional_scripts)
    # optional always included — never drop a listed script

    # NEVER skip training or any listed script. include_all_models is always on.
    if config.get("skip_training_in_pipeline"):
        logging.warning(
            "skip_training_in_pipeline was True in config — IGNORING (user: don't skip anything)"
        )
    logging.info("Running ALL %s scripts (training + optional included=%s)", len(scripts_to_run), args.include_optional)

    failed = []
    skipped = []
    succeeded = []
    total = len(scripts_to_run)
    _banner(f"PIPELINE QUEUE: {total} scripts (show_output={show_output})")
    for i, script in enumerate(scripts_to_run, start=1):
        print(f"\n>>> next [{i}/{total}]: {script}", flush=True)
        status = "fail"
        for attempts in range(1, args.retry + 1):
            if attempts > 1:
                print(
                    f"  retry {attempts}/{args.retry} for {script} ...",
                    flush=True,
                )
                logging.warning(
                    "Retrying %s (attempt %s/%s)", script, attempts, args.retry
                )
            status = run_script(
                script,
                config,
                index=i,
                total=total,
                show_output=show_output,
            )
            if status in ("ok", "skip"):
                break
            time.sleep(1)
        if status == "ok":
            succeeded.append(script)
            print(f">>> result [{i}/{total}] OK  {script}", flush=True)
        elif status == "skip":
            # skip is not allowed silently — count as fail
            failed.append(script)
            print(f">>> result [{i}/{total}] FAIL-SKIP {script} (skips are errors)", flush=True)
            if args.strict:
                logging.error("Pipeline aborted on skipped %s", script)
                sys.exit(1)
        else:
            failed.append(script)
            print(f">>> result [{i}/{total}] FAIL {script}", flush=True)
            if args.strict:
                logging.error("Pipeline aborted on %s", script)
                sys.exit(1)

    _banner(
        f"PIPELINE SUMMARY  ok={len(succeeded)}  skip={len(skipped)}  fail={len(failed)}"
    )
    if succeeded:
        print("  OK:", ", ".join(succeeded), flush=True)
    if skipped:
        print("  SKIP:", ", ".join(skipped), flush=True)
    if failed:
        print("  FAIL:", ", ".join(failed), flush=True)
        logging.warning("Failures: %s", ", ".join(failed))

    logging.info(
        "Pipeline summary: ok=%s skip=%s fail=%s",
        len(succeeded),
        len(skipped),
        len(failed),
    )
    if failed:
        if not continue_on_error:
            sys.exit(2)
        sys.exit(0 if succeeded else 2)

    if og_snap:
        try:
            from fxjefe_paths import verify_og_models_untouched
            mutated = verify_og_models_untouched(og_snap)
        except Exception as e:
            mutated = [str(e)]
        if mutated:
            logging.error("OG MODELS CHANGED (should never happen): %s", mutated)
            print("OG MODEL GUARD FAIL:", mutated, flush=True)
            sys.exit(3)
        logging.info("OG model guard: all %s original model files unchanged", len(og_snap))
        print(f"OG model guard: {len(og_snap)} original models untouched", flush=True)

    logging.info("Pipeline completed successfully")
    print("Pipeline completed successfully", flush=True)


if __name__ == "__main__":
    main()
