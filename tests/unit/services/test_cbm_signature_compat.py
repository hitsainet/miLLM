"""Our ContinuousBatchingManager call must match the INSTALLED transformers.

WHAT HAPPENED (2026-07-27)
cbm_backend called ContinuousBatchingManager(..., max_queue_size=N). The dev
venv had transformers 5.0.0, where that is a real parameter, so every test
passed. The image built `transformers>=4.47.0` unpinned and shipped 5.14.1,
where the parameter moved into a ContinuousBatchingConfig object. Production
logged

    cbm_start_failed: ContinuousBatchingManager.__init__() got an unexpected
    keyword argument 'max_queue_size'

and continuous batching silently never ran — millm_status reported
cbm_enabled:true / cbm_running:false, which reads as "on" at a glance.

WHY THIS TEST IS SHAPED THIS WAY
It introspects the signature of whatever transformers is actually installed and
asserts our construction is valid against IT. A test that hardcoded either
calling convention would keep passing on the version it was written for — which
is exactly the failure being prevented. CI installs from pyproject, so CI sees
the same version the image will.

MUTATION CONTROLS:
  * hardcode max_queue_size= again        -> compat test fails on 5.14+
  * drop the config-object branch          -> compat test fails on 5.14+
  * remove the signature introspection     -> adaptive test fails
"""

import inspect

import pytest

transformers = pytest.importorskip("transformers")


def _manager_params():
    from transformers import ContinuousBatchingManager

    return inspect.signature(ContinuousBatchingManager.__init__).parameters


class TestWeBuildTheCallFromTheRealSignature:
    def test_backend_introspects_rather_than_assumes(self):
        import millm.services.cbm_backend as mod

        src = inspect.getsource(mod)
        assert "inspect.signature(ContinuousBatchingManager.__init__)" in src, (
            "cbm_backend assumes a calling convention instead of reading the "
            "installed one — the exact defect that broke continuous batching"
        )

    def test_both_known_conventions_are_handled(self):
        import millm.services.cbm_backend as mod

        src = inspect.getsource(mod)
        assert "continuous_batching_config" in src, (
            "the transformers >= 5.14 config-object form is not handled"
        )
        assert '"max_queue_size" in _params' in src, (
            "the transformers <= 5.0 kwarg form is not handled"
        )

    def test_the_selected_kwargs_are_accepted_by_this_transformers(self):
        """The real check: would our call actually bind?"""
        params = _manager_params()

        if "continuous_batching_config" in params:
            chosen = {"model", "generation_config", "continuous_batching_config"}
        elif "max_queue_size" in params:
            chosen = {"model", "generation_config", "max_queue_size"}
        else:
            chosen = {"model", "generation_config"}

        unknown = chosen - set(params)
        assert not unknown, (
            f"cbm_backend would pass {sorted(unknown)}, which transformers "
            f"{transformers.__version__} does not accept"
        )

    def test_max_queue_size_is_reachable_somewhere(self):
        """It must land SOMEWHERE, or the queue bound is silently ignored."""
        params = _manager_params()

        if "max_queue_size" in params:
            return  # direct kwarg form

        assert "continuous_batching_config" in params, (
            f"transformers {transformers.__version__} exposes neither "
            "max_queue_size nor continuous_batching_config; the queue bound "
            "cannot be set at all"
        )
        from transformers.generation.configuration_utils import (
            ContinuousBatchingConfig,
        )

        cfg_params = inspect.signature(ContinuousBatchingConfig.__init__).parameters
        assert "max_queue_size" in cfg_params, (
            "ContinuousBatchingConfig no longer takes max_queue_size"
        )


class TestTheUnpinnedDependencyIsAcknowledged:
    def test_the_drift_hazard_is_documented(self):
        import millm.services.cbm_backend as mod

        src = inspect.getsource(mod)
        assert "transformers>=4.47.0" in src or ">=4.47" in src, (
            "the unpinned transformers floor is what allows this API to change "
            "under the image; say so where the workaround lives"
        )
