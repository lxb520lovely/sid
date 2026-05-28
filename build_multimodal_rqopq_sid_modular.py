#!/usr/bin/env python3
"""Modular entrypoint for the multi-modal RQ-OPQ SID builder.

The original build_multimodal_rqopq_sid.py is left untouched. This script uses
the same arguments and behavior, with implementation split under
multimodal_rqopq_sid_code/.
"""

from __future__ import annotations

from multimodal_rqopq_sid_code.config import parse_args
from multimodal_rqopq_sid_code.pipeline import run


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
