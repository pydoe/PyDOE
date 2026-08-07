"""Debug utilities for PyDOE: environment introspection for bug reports."""

import argparse
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _get_version(package_name: str) -> str:
    """Get package version or 'unknown' if not found.

    Parameters
    ----------
    package_name : str
        Name of the package.

    Returns
    -------
    str
        Version string or 'unknown' if not found.
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _detect_install_source() -> str:
    """Detect whether pydoe was installed via conda, pypi, or editable.

    Returns
    -------
    str
        One of: "conda", "editable (dev)", or "pypi".
    """
    conda_meta_path = os.path.join(sys.prefix, "conda-meta")
    if os.path.isdir(conda_meta_path):
        return "conda"

    # Check for editable installation
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / py_ver / "site-packages"
    pydoe_dist_info = site_packages.glob("pydoe-*.dist-info")
    for dist_info in pydoe_dist_info:
        direct_url_path = dist_info / "direct_url.json"
        if direct_url_path.exists():
            try:
                with open(direct_url_path, encoding="utf-8") as f:
                    direct_url = json.load(f)
                    if direct_url.get("editable"):
                        return "editable (dev)"
            except (json.JSONDecodeError, OSError):
                pass

    return "pypi"


def debug_info() -> dict:
    """Collect environment and package information for bug reports.

    Returns
    -------
    dict
        Dictionary with keys: pydoe, python, python_impl, os, os_release,
        architecture, numpy, scipy, install_source.
    """
    return {
        "pydoe": _get_version("pydoe"),
        "python": f"{sys.version.split()[0]}",
        "python_impl": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "numpy": _get_version("numpy"),
        "scipy": _get_version("scipy"),
        "install_source": _detect_install_source(),
    }


def format_debug_output(info: dict) -> str:
    """Format debug info as a markdown table.

    Parameters
    ----------
    info : dict
        Dictionary from debug_info().

    Returns
    -------
    str
        Markdown-formatted table, ready to paste into GitHub issues.
    """
    lines = ["### Debug Info", "", "| Field | Value |", "|-------|-------|"]

    field_names = {
        "pydoe": "PyDOE",
        "python": "Python",
        "python_impl": "Python Implementation",
        "os": "OS",
        "os_release": "OS Release",
        "architecture": "Architecture",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "install_source": "Install Source",
    }

    for key in [
        "pydoe",
        "python",
        "python_impl",
        "os",
        "os_release",
        "architecture",
        "numpy",
        "scipy",
        "install_source",
    ]:
        lines.append(f"| {field_names[key]} | `{info[key]}` |")

    return "\n".join(lines)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PyDOE utilities", prog="pydoe"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version('pydoe')}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for bug reports",
    )

    args = parser.parse_args()

    if args.debug:
        info = debug_info()
        output = format_debug_output(info)
        print(output)
    elif not sys.argv[1:]:
        parser.print_help()
