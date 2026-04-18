#!/usr/bin/env python3
"""Run project tests without auto-loading global pytest plugins (e.g. ROS launch_testing)."""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    import pytest

    args = sys.argv[1:] if len(sys.argv) > 1 else ["tests"]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
