"""Single entry point: `srm <command> [options]`.

Declared in pyproject.toml as the `srm` console script. Every subcommand is a
thin delegation to the module that already owns that step, so there is exactly
one implementation of each and `srm train ...` and `python -m srm.train ...`
cannot drift apart.

    srm fetch-sen2neon --limit 150              # real 10 m / 2.5 m pairs
    srm prepare-sen2neon --out data/patches_sen2neon
    srm overfit --config configs/sen2neon_x4.yaml
    srm train --config configs/sen2neon_x4.yaml
    srm eval  --checkpoint checkpoints/sen2neon_x4/best.pt --split test
    srm infer --checkpoint ... --input scene.tif --output sr.tif
    srm ensemble --checkpoints a.pt b.pt --input scene.tif --outdir unc/
    srm app                                     # Streamlit dashboard

    srm demo                                    # synthetic data, no download

Imports are deferred into each handler: torch costs seconds to import, and
`srm --help` should not pay for it.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable

from . import __version__

# Subcommands that live in scripts/ rather than in the package. They are part
# of the repository, not of the installed wheel, because they are operational
# tools rather than library code.
_SCRIPTS = {
    "demo": ("run_demo.py", "run the full pipeline end to end on synthetic scenes"),
    "prepare": ("prepare_dataset.py", "build a paired patch dataset"),
    "download": ("download_copernicus.py", "fetch Sentinel-2 L2A from Copernicus Data Space"),
    "fetch-sen2neon": ("download_sen2neon.py",
                       "download real Sentinel-2/NEON 10 m-2.5 m pairs"),
    "prepare-sen2neon": ("prepare_sen2neon.py",
                         "build a geographically split dataset from SEN2NEON"),
    "overfit": ("overfit_check.py", "overfit a tiny batch to verify the training path"),
    "check": ("check_install.py", "verify this checkout works on this machine"),
    "stamp": ("stamp_weights.py", "pick raw or EMA weights for a checkpoint by measurement"),
}

_MODULES = {
    "train": ("srm.train", "train a model from a YAML config"),
    "eval": ("srm.evaluate", "score a checkpoint against the bicubic baseline"),
    "infer": ("srm.infer", "super-resolve a GeoTIFF scene"),
    "uncertainty": ("srm.uncertainty", "single-model reference-free uncertainty maps"),
    "ensemble": ("srm.ensemble",
                 "disagreement across independently trained checkpoints"),
}


def _repo_root() -> Path:
    """The checkout this package was imported from, if it is one."""
    return Path(__file__).resolve().parents[1]


def _load_script(filename: str) -> Callable[[list[str] | None], None]:
    """Import a scripts/ file by path and hand back its `main`.

    Searched next to the package first, then in the working directory, so this
    works from a source checkout and from `pip install -e .`. A non-editable
    install has no scripts/ directory, and the error says so rather than
    surfacing a bare ImportError.
    """
    for base in (_repo_root(), Path.cwd()):
        path = base / "scripts" / filename
        if path.is_file():
            spec = importlib.util.spec_from_file_location(f"_srm_script_{path.stem}", path)
            if spec is None or spec.loader is None:  # pragma: no cover
                break
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module.main
    raise SystemExit(
        f"cannot find scripts/{filename}.\n"
        f"Looked in {_repo_root() / 'scripts'} and {Path.cwd() / 'scripts'}.\n"
        "This subcommand needs the source checkout; run it from the repository "
        "root, or install with `pip install -e .`."
    )


def _run_app(argv: list[str]) -> None:
    """Hand off to Streamlit, which insists on owning the process."""
    app = _repo_root() / "app" / "app.py"
    if not app.is_file():
        raise SystemExit(f"{app} not found; the viewer needs the source checkout.")
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:
        raise SystemExit(f"streamlit is not installed ({exc}). pip install streamlit") from exc
    sys.argv = ["streamlit", "run", str(app), *argv]
    sys.exit(stcli.main())


def _list_models(_: list[str]) -> None:
    from .models.registry import REGISTRY

    width = max(len(name) for name in REGISTRY)
    print("Available model names for `model.name` in a config:\n")
    for name in sorted(REGISTRY):
        doc = (REGISTRY[name].__doc__ or "").strip().splitlines()
        print(f"  {name:<{width}}  {doc[0] if doc else ''}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srm",
        description="Super-Resolution Mapping of medium-resolution satellite imagery.",
        epilog="Run `srm <command> --help` for a command's own options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"srm {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, (_, help_text) in {**_SCRIPTS, **_MODULES}.items():
        sub.add_parser(name, help=help_text, add_help=False)
    sub.add_parser("app", help="launch the Streamlit comparison viewer", add_help=False)
    sub.add_parser("models", help="list the available architectures", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # Split on the subcommand rather than letting argparse recurse: each
    # delegate owns a complete parser of its own, including its --help.
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return
    if argv[0] == "--version":
        parser.parse_args(argv)
        return

    command, rest = argv[0], argv[1:]

    # Give the delegate the argv it would have seen if it were the program, so
    # anything reading sys.argv directly sees its own arguments and not ours.
    # (This does not rename the usage line under `python -m srm.cli`: since
    # 3.14 argparse derives `prog` from sys.orig_argv for -m invocations. Under
    # the installed `srm` console script it reads correctly.)
    sys.argv = [f"{parser.prog} {command}", *rest]

    if command in _MODULES:
        import importlib

        importlib.import_module(_MODULES[command][0]).main(rest)
    elif command in _SCRIPTS:
        _load_script(_SCRIPTS[command][0])(rest)
    elif command == "app":
        _run_app(rest)
    elif command == "models":
        _list_models(rest)
    else:
        parser.print_help(sys.stderr)
        raise SystemExit(f"\nunknown command '{command}'")


if __name__ == "__main__":
    main()
