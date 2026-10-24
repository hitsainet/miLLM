"""miLLM runs the CUDA allocator with expandable segments (2026-10-05).

Without them an unload after any request left the whole model RESERVED (8 MiB allocated, 17,080
MiB reserved on JEV-9B), so the card stayed full with nothing loaded.
"""

import os
import subprocess
import sys

from millm.core.cuda_allocator import NEW_NAME, OLD_NAME, allocator_env


def test_unset_gets_expandable_segments_on_the_name_every_torch_reads():
    assert allocator_env({}) == (OLD_NAME, "expandable_segments:True")


def test_other_options_are_kept_and_extended():
    assert allocator_env({OLD_NAME: "max_split_size_mb:128"}) == (
        OLD_NAME, "max_split_size_mb:128,expandable_segments:True")


def test_an_explicit_choice_is_the_operators():
    assert allocator_env({OLD_NAME: "expandable_segments:False"}) == (OLD_NAME, "expandable_segments:False")


def test_the_newer_variable_is_edited_when_it_is_the_one_set():
    assert allocator_env({NEW_NAME: "garbage_collection_threshold:0.8"}) == (
        NEW_NAME, "garbage_collection_threshold:0.8,expandable_segments:True")


def test_importing_millm_sets_it_before_torch_can_read_it():
    """The WIRING: importing the package applies the setting, in a fresh interpreter, before torch
    is imported. A unit test of `allocator_env` alone passes with the call deleted."""
    env = {k: v for k, v in os.environ.items() if k not in (OLD_NAME, NEW_NAME)}
    code = (
        "import sys, os, millm; "
        "assert 'torch' not in sys.modules, 'millm imported torch before configuring it'; "
        f"print(os.environ.get('{OLD_NAME}'))"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "expandable_segments:True"
