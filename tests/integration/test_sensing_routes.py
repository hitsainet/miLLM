"""
Feature 11 Task 4.2: /api/sensing route tests — envelope shapes, filters,
enable/disable toggling the column + live arm state.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from millm.main import create_app


@pytest.fixture
def client():
    import millm.api.dependencies as deps

    deps._sensing_service = None  # fresh singleton per test
    app = create_app()
    with TestClient(app) as tc:
        yield tc
    deps._sensing_service = None


class TestStatusRoute:
    def _ctx(self, rows=()):
        session = MagicMock()

        async def _execute(*a, **kw):
            result = MagicMock()
            result.__iter__ = lambda self: iter(rows)
            return result

        session.execute = _execute

        class _Ctx:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    def test_status_envelope_unarmed(self, client):
        with patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()):
            response = client.get("/api/sensing/status")
        body = response.json()
        assert body["success"] is True
        assert body["data"]["armed"] is False
        assert body["data"]["retention"]["max_events_per_cluster"] > 0
        assert body["data"]["enabled_clusters"] == []

    def test_status_reports_enabled_but_unarmed_cluster(self, client):
        """FTID pitfall 8: persistent intent visible distinctly from armed."""
        row = MagicMock(id="prof_s1", is_active=False)
        row.name = "fear cluster"  # name= is a reserved MagicMock kwarg
        with patch("millm.db.base.async_session_factory",
                   return_value=self._ctx(rows=(row,))):
            response = client.get("/api/sensing/status")
        data = response.json()["data"]
        assert data["armed"] is False
        assert data["enabled_clusters"] == [
            {"id": "prof_s1", "name": "fear cluster", "is_active": False}
        ]


class TestEventRoutes:
    def _session_ctx(self, repo_result=None, count=0, single=None):
        session = MagicMock()

        async def _commit():
            return None

        session.commit = _commit

        class _Ctx:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    def test_list_events_envelope(self, client):
        from millm.db.models.sensing_event import SensingEvent

        event = SensingEvent(
            id=1, profile_id="prof_s1", request_id="req-1", phase="decode",
            pos_start=5, pos_end=6, fired_members=[[7, 8.4]], fired_count=1,
            score=2.1, summary="fear: 1/3 members fired", truncated=False,
        )
        with patch("millm.db.repositories.sensing_repository.SensingRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._session_ctx()):
            instance = MockRepo.return_value
            instance.list_events = _async_return([event])
            instance.count = _async_return(1)
            instance.prune_aged = _async_return(0)
            response = client.get("/api/sensing/events?profile_id=prof_s1&limit=10")
        body = response.json()
        assert body["success"] is True
        assert body["data"]["total"] == 1
        assert body["data"]["events"][0]["summary"].startswith("fear:")
        kwargs = instance.list_events.call_args.kwargs
        assert kwargs["profile_id"] == "prof_s1" and kwargs["limit"] == 10

    def test_event_detail_includes_context(self, client):
        from millm.db.models.sensing_event import SensingEvent

        event = SensingEvent(
            id=7, profile_id="p", request_id="r", phase="decode",
            pos_start=1, pos_end=1, fired_members=[], fired_count=2,
            score=1.0, summary="s", truncated=False,
            context_text="the deep ocean", context_token_ids=[1, 2, 3],
        )
        with patch("millm.db.repositories.sensing_repository.SensingRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._session_ctx()):
            MockRepo.return_value.get = _async_return(event)
            response = client.get("/api/sensing/events/7")
        assert response.json()["data"]["context_text"] == "the deep ocean"

    def test_event_detail_missing_is_404(self, client):
        """Pruned events are EXPECTED under retention — clients branch on
        404 (011 R1 fix; this pin previously asserted only the envelope)."""
        with patch("millm.db.repositories.sensing_repository.SensingRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._session_ctx()):
            MockRepo.return_value.get = _async_return(None)
            response = client.get("/api/sensing/events/999")
        assert response.status_code == 404
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "SENSING_EVENT_NOT_FOUND"
        assert "not found" in body["error"]["message"]

    def test_clear_events(self, client):
        with patch("millm.db.repositories.sensing_repository.SensingRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._session_ctx()):
            MockRepo.return_value.clear = _async_return(5)
            response = client.delete("/api/sensing/events?profile_id=prof_s1")
        assert response.json()["data"]["deleted"] == 5


class TestToggleRoutes:
    def _profile(self, source_kind="cluster", is_active=False):
        profile = MagicMock()
        profile.id = "prof_s1"
        profile.source_kind = source_kind
        profile.is_active = is_active
        profile.sensing_enabled = False
        profile.cluster_meta = {
            "members": [{"feature_idx": 7, "strength": 1.0,
                         "max_activation": 40.0},
                        {"feature_idx": 9, "strength": 1.0,
                         "max_activation": 20.0}],
        }
        profile.name = "fear cluster"
        return profile

    def _ctx(self):
        session = MagicMock()

        async def _commit():
            return None

        session.commit = _commit

        class _Ctx:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    def test_enable_persists_column(self, client):
        profile = self._profile()
        with patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()):
            MockRepo.return_value.get = _async_return(profile)
            response = client.post("/api/sensing/prof_s1/enable")
        body = response.json()
        assert body["success"] is True
        assert profile.sensing_enabled is True
        assert body["data"]["sensing_enabled"] is True
        assert body["data"]["armed"] is False  # not the active cluster

    def test_enable_on_active_cluster_live_arms(self, client):
        profile = self._profile(is_active=True)
        sae = MagicMock()
        sae.d_sae = 16384
        with patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()), \
             patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockRepo.return_value.get = _async_return(profile)
            MockState.return_value.attached_sae = sae
            response = client.post("/api/sensing/prof_s1/enable")
        assert response.json()["data"]["armed"] is True
        sae.arm_sensing.assert_called_once()

    def test_disable_on_active_cluster_live_disarms(self, client):
        profile = self._profile(is_active=True)
        profile.sensing_enabled = True
        sae = MagicMock()
        with patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()), \
             patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockRepo.return_value.get = _async_return(profile)
            MockState.return_value.attached_sae = sae
            response = client.post("/api/sensing/prof_s1/disable")
        assert response.json()["data"]["armed"] is False
        sae.disarm_sensing.assert_called_once()

    def test_manual_profile_refused(self, client):
        profile = self._profile(source_kind="manual")
        with patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()):
            MockRepo.return_value.get = _async_return(profile)
            response = client.post("/api/sensing/prof_s1/enable")
        body = response.json()
        assert body["success"] is False
        assert "clusters only" in body["error"]["message"]

    def test_unknown_profile_404_style(self, client):
        with patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()):
            MockRepo.return_value.get = _async_return(None)
            response = client.post("/api/sensing/ghost/enable")
        assert response.json()["success"] is False


def _async_return(value):
    async def _call(*args, **kwargs):
        return value

    return MagicMock(side_effect=_call)


class TestConfigRoute:
    """Goal item 4: runtime min_k override — persisted locally (export
    stays lossless) with live re-arm."""

    def _profile(self, is_active=False):
        profile = MagicMock()
        profile.id = "prof_s1"
        profile.source_kind = "cluster"
        profile.is_active = is_active
        profile.sensing_enabled = True
        profile.cluster_meta = {
            "members": [
                {"feature_idx": 7, "strength": 1.0, "max_activation": 40.0},
                {"feature_idx": 9, "strength": 1.0, "max_activation": 20.0},
                {"feature_idx": 12, "strength": 1.0, "max_activation": 10.0},
            ],
        }
        profile.name = "fear cluster"
        return profile

    def _ctx(self):
        session = MagicMock()

        async def _commit():
            return None

        session.commit = _commit

        class _Ctx:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    def test_set_min_k_persists_local_override(self, client):
        profile = self._profile()
        with patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()):
            MockRepo.return_value.get = _async_return(profile)
            response = client.put("/api/sensing/prof_s1/config",
                                  json={"min_k": 2})
        body = response.json()
        assert body["success"] is True
        assert body["data"]["effective_min_k"] == 2
        # stored OUTSIDE the portable document
        assert profile.cluster_meta["sensing_overrides"] == {"min_k": 2}
        assert "min_k" not in profile.cluster_meta.get("sensing", {})

    def test_null_clears_override_back_to_all(self, client):
        profile = self._profile()
        profile.cluster_meta["sensing_overrides"] = {"min_k": 1}
        with patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()):
            MockRepo.return_value.get = _async_return(profile)
            response = client.put("/api/sensing/prof_s1/config",
                                  json={"min_k": None})
        body = response.json()
        assert body["success"] is True
        assert body["data"]["effective_min_k"] == 3  # ALL sensable members
        assert "sensing_overrides" not in profile.cluster_meta

    def test_out_of_range_min_k_refused(self, client):
        profile = self._profile()
        with patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo, \
             patch("millm.db.base.async_session_factory",
                   return_value=self._ctx()):
            MockRepo.return_value.get = _async_return(profile)
            response = client.put("/api/sensing/prof_s1/config",
                                  json={"min_k": 9})
        body = response.json()
        assert body["success"] is False
        assert "between 1 and 3" in body["error"]["message"]
