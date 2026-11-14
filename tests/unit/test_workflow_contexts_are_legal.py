"""F20 R3-04. A workflow file GitHub rejects runs NOTHING — silently.

`HF_HOME: ${{ runner.temp }}/hf` in a job-level `env:` block is a schema
violation: the `runner` context exists only inside steps. GitHub does not run
the job and report an error — it rejects the entire WORKFLOW FILE, so every
job in it, including the pre-existing backend test suite, quietly stops
running. The Actions UI says only "This run likely failed because of a workflow
file issue", with no logs and no jobs.

Four commits shipped that way while I reported the suite green from local runs.
The failure is invisible from the terminal: `pytest` passes, `git push`
succeeds, and nothing in the working copy is wrong.

This is cheap to prevent and expensive to notice, so it is a test.
"""

import re
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

#: Contexts legal in a JOB-LEVEL `env:` block.
#: https://docs.github.com/actions/learn-github-actions/contexts#context-availability
JOB_ENV_CONTEXTS = {"github", "vars", "secrets", "inputs", "needs", "matrix", "strategy"}

#: Contexts legal in a job's `if:` — notably NOT `runner` or `env`.
JOB_IF_CONTEXTS = JOB_ENV_CONTEXTS | {"always", "success", "failure", "cancelled"}


def _workflow_files():
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _contexts(value: str) -> set[str]:
    return set(re.findall(r"\$\{\{\s*([a-zA-Z_]+)\s*\.", str(value)))


@pytest.mark.skipif(yaml is None, reason="pyyaml not installed")
@pytest.mark.skipif(not WORKFLOWS.is_dir(), reason=f"no workflows at {WORKFLOWS}")
class TestWorkflowContextsAreLegal:
    def test_the_extraction_works(self):
        """An empty file list passes every assertion below it."""
        files = _workflow_files()
        assert files, f"no workflow files found under {WORKFLOWS}"
        parsed = [f for f in files if yaml.safe_load(f.read_text())]
        assert parsed, "no workflow file parsed to anything"

    def test_no_job_level_env_uses_an_out_of_scope_context(self):
        """The exact defect: `runner` (or `env`, or `steps`) in a job `env:`."""
        offences = []
        for f in _workflow_files():
            doc = yaml.safe_load(f.read_text()) or {}
            for jname, job in (doc.get("jobs") or {}).items():
                if not isinstance(job, dict):
                    continue
                for key, value in (job.get("env") or {}).items():
                    illegal = _contexts(value) - JOB_ENV_CONTEXTS
                    for ctx in sorted(illegal):
                        offences.append(
                            f"{f.name}: jobs.{jname}.env.{key} uses "
                            f"${{{{ {ctx}.* }}}}"
                        )
        assert not offences, (
            "Out-of-scope context(s) in a job-level env: block:\n  "
            + "\n  ".join(offences)
            + "\n\nGitHub rejects the WHOLE WORKFLOW FILE for this — no job "
            "runs, no logs are produced, and the only signal is 'this run "
            "likely failed because of a workflow file issue'. Move the "
            "reference into a step's env:, or hardcode the value."
        )

    def test_no_job_level_if_uses_an_out_of_scope_context(self):
        """Same class, adjacent scope — `runner`/`env` are illegal here too."""
        offences = []
        for f in _workflow_files():
            doc = yaml.safe_load(f.read_text()) or {}
            for jname, job in (doc.get("jobs") or {}).items():
                if not isinstance(job, dict) or "if" not in job:
                    continue
                illegal = _contexts(job["if"]) - JOB_IF_CONTEXTS
                for ctx in sorted(illegal):
                    offences.append(f"{f.name}: jobs.{jname}.if uses ${{{{ {ctx}.* }}}}")
        assert not offences, (
            "Out-of-scope context(s) in a job-level if::\n  " + "\n  ".join(offences)
        )

    def test_every_workflow_still_parses(self):
        """A YAML error has the same blast radius and is even quieter."""
        for f in _workflow_files():
            try:
                yaml.safe_load(f.read_text())
            except yaml.YAMLError as exc:
                pytest.fail(f"{f.name} is not valid YAML — nothing in it will run: {exc}")
