"""Rev2 review of 040 P0 (2026-11-08): forms in which a `vitest:` exercise never runs, or runs
skipped, while T03's textual `vitest_problem` accepts it. Each body below was run under the admin
UI's own Vitest (5.0.5, node:26, CI=true): the file passed and the test named "the name" was
`skipped`, `todo` or not reported at all.
"""

from __future__ import annotations

import pytest

from tests.unit.legacy.test_frozen_identifiers import vitest_problem

HEAD = "import { describe, expect, it, test } from 'vitest';\nit('anchor', () => { expect(1).toBe(1); });\n"

SLIPS = {
    "describe.skipIf": "describe.skipIf(true)('x', () => { it('the name', () => {}); });",
    "describe.runIf": "describe.runIf(false)('x', () => { it('the name', () => {}); });",
    "describe.concurrent.skip": "describe.concurrent.skip('x', () => { it('the name', () => {}); });",
    "describe.skip.each": "describe.skip.each([1])('x %i', () => { it('the name', () => {}); });",
    "line comment": "// it('the name', () => {});",
    "block comment": "/* it('the name', () => {}); */",
    "string literal": "const src = \"it('the name', () => {})\";",
    "options skip": "it('the name', { skip: true }, () => {});",
    "options todo": "it('the name', { todo: true }, () => {});",
    "context skip": "it('the name', (ctx) => { ctx.skip(); });",
}


@pytest.mark.parametrize("form", sorted(SLIPS))
def test_a_never_running_vitest_exercise_is_refused(tmp_path, form):
    file = tmp_path / "x.test.ts"
    file.write_text(HEAD + SLIPS[form] + "\n")
    assert vitest_problem(file, "the name") is not None, f"T03 accepts an exercise that never runs: {form}"


def test_a_running_exercise_is_still_accepted(tmp_path):
    file = tmp_path / "x.test.ts"
    file.write_text(HEAD + "describe('x', () => { it('the name', () => { expect(1).toBe(1); }); });\n")
    assert vitest_problem(file, "the name") is None
