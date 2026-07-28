"""Stable command-line surface for portable post-training projects."""

from __future__ import annotations

import shutil  # noqa: F401 - exposed for test monkeypatching via posttrain_cli.cli.shutil
import subprocess
import sys
import traceback
from collections.abc import Sequence

import click
from posttrain.common import ContractError

from .app import create_app
from .errors import error_message

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else None
    argv_list = args if args is not None else sys.argv
    json_output = "--json" in argv_list
    want_traceback = "--traceback" in argv_list
    json_stream = sys.stdout
    if json_output and argv is None:
        sys.stdout = sys.stderr

    app = create_app(json_stream=json_stream)
    try:
        result = app(args=args, standalone_mode=False)
        return int(result) if isinstance(result, int) else 0
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.ClickException as exc:
        exc.show()
        return int(exc.exit_code)
    except click.Abort:
        return 1
    except (
        ContractError,
        FileExistsError,
        KeyError,
        LookupError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        if want_traceback:
            traceback.print_exc(file=sys.stderr)
        print(f"error: {error_message(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
