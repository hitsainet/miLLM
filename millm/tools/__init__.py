"""Operator command-line tools that ship inside the `millm` package.

They live here rather than under `scripts/` because the public mirror strips `scripts/`; anything an
operator of the published image is told to run has to be importable as `python -m millm.tools.<x>`.
"""
