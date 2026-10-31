"""No pattern text reaches standard error (039 FTASKS 11.3; FTDD §6.6).

google-re2 writes a pattern it cannot parse to stderr unless `log_errors` is off. Patterns are
operator data and stderr is a log, so a pattern the scanner did not foresee must reach RE2 with
logging off. The scanner is bypassed here so an unparseable pattern reaches `re2.compile`.

Control (039 controls record): `options.log_errors = True` → the pattern appears on fd 2 → red.
"""

from __future__ import annotations

import pytest

from millm.rules import dialect

PATTERN = "qzv(unbalanced"


def test_an_re2_refusal_is_typed_and_silent(capfd, monkeypatch):
    monkeypatch.setattr(dialect, "check", lambda pattern: None)
    dialect.compile_pattern.cache_clear()
    with pytest.raises(dialect.PatternRefused) as caught:
        dialect.compile_pattern(PATTERN, False)
    assert caught.value.rule == "syntax"
    err = capfd.readouterr().err
    assert "qzv" not in err and "unbalanced" not in err, err


def test_the_option_name_exists_in_the_pinned_binding():
    import re2

    options = re2.Options()
    options.log_errors = False
    assert options.log_errors is False
