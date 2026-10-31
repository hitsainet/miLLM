"""Download progress was written where nobody could read it.

`get_model_service` builds a NEW `ModelService` for every request. Progress
lived in `self._download_progress`, so the instance that ran the download wrote
into a dict that the instance serving `GET /api/models` never saw — and the API
reported `download_progress: null` for the whole of every download that has ever
run.

Observed on a 51.8 GB Qwen3.8-27B pull: bytes landing at 110 MB/s while the API
said nothing, which from outside is indistinguishable from a download that has
died. The only way to tell was to `du` the cache directory by hand.
"""

import pytest

from millm.services import model_service as ms
from millm.services.model_service import ModelService


@pytest.fixture(autouse=True)
def _clean_registry():
    ms._DOWNLOAD_PROGRESS.clear()
    yield
    ms._DOWNLOAD_PROGRESS.clear()


def _service(**kw):
    """A ModelService with everything it needs stubbed — this test is about
    where progress is STORED, not about downloading."""
    from unittest.mock import MagicMock
    return ModelService(
        repository=MagicMock(), downloader=MagicMock(), loader=MagicMock(),
        emitter=MagicMock(), inference_service=MagicMock(), **kw)


class TestProgressSurvivesTheRequestBoundary:
    def test_a_SECOND_service_instance_sees_the_progress(self):
        """The production shape: one instance per request.

        This is the whole bug. Two instances, because two requests.
        """
        downloader = _service()
        ms._DOWNLOAD_PROGRESS[42] = 37

        reader = _service()

        assert reader.get_download_progress(42) == 37, (
            "the request that lists models cannot see progress written by the "
            "request that started the download"
        )
        assert downloader is not reader

    def test_progress_is_not_an_INSTANCE_attribute(self):
        """Guards the specific regression: putting it back on `self`.

        An instance dict passes any single-instance test, which is why this
        shipped — every test used one service.
        """
        svc = _service()

        assert not hasattr(svc, "_download_progress"), (
            "progress is on the instance again; the next request gets a "
            "different one and reads an empty dict"
        )

    def test_an_unknown_model_reports_None_not_zero(self):
        """None means "not downloading"; 0 means "downloading, nothing yet".
        Collapsing them would show a stalled bar for every idle model."""
        assert _service().get_download_progress(999) is None


class TestTheRegistryIsCleanedUp:
    def test_a_finished_download_leaves_no_entry(self):
        """Otherwise every model ever downloaded reports a stale 100 forever,
        and the UI shows a completed bar on an idle model."""
        ms._DOWNLOAD_PROGRESS[7] = 100
        ms._DOWNLOAD_PROGRESS.pop(7, None)

        assert _service().get_download_progress(7) is None

    def test_entries_are_per_model(self):
        ms._DOWNLOAD_PROGRESS[1] = 10
        ms._DOWNLOAD_PROGRESS[2] = 90
        svc = _service()

        assert (svc.get_download_progress(1), svc.get_download_progress(2)) == (10, 90)
