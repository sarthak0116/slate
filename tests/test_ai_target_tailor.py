"""AI tailoring against user-pasted job targets (not just the sample catalog).

The deterministic target flow deliberately preserves the original resume. The
opt-in AI path rewrites, so it keeps every safeguard of the sample-job AI path:
grounded prompt, mechanical citation re-checks, approval gate. No network:
MockLLM stands in for the provider. Run: python -m pytest tests/test_ai_target_tailor.py -q
"""
import os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
pytest.importorskip("fastapi"); pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from slate_core.ai_service import AISlateService
from slate_core.llm import MockLLM
from slate_core.service import SlateError
from slate_api.app import create_authenticated_app
from slate_core.auth import DevAuth
from slate_core.user_store import MemUserStore, SqlUserStore

RESUME = ("Asha Verma\nB.Tech Computer Science, IIITM Gwalior, 2027\n"
          "Intern at ExampleCo (3 months)\nSkills: Python, Git, Excel\n")
JD = "Backend engineer needed: Python, Git. Internship experience a plus."
H = {"Authorization": "Bearer dev:asha@example.com"}

GROUNDED_REPLY = (
    "RESUME\n"
    "Asha Verma [S1]\n"
    "B.Tech Computer Science, IIITM Gwalior, 2027 [S2]\n"
    "Intern at ExampleCo (3 months) [S3]\n"
    "Skills: Python, Git, Excel [S4]\n"
    "COVER NOTE\n"
    "I am applying for the Backend engineer role [J]. "
    "My Python and Git skills [S4] match the posting [J].")


def _seeded(svc):
    rid = svc._upload_resume(RESUME)["resumeId"]
    tid = svc.create_target("Backend Engineer", "Zeta", JD)["target"]["targetId"]
    return rid, tid


def test_ai_target_draft_happy_path_grounded():
    llm = MockLLM(reply=GROUNDED_REPLY)
    svc = AISlateService(llm_client=llm)
    rid, tid = _seeded(svc)
    out = svc.tailor_ai_target(rid, tid)
    assert out["status"] == "draft" and out["requiresApproval"] is True
    assert out["aiGenerated"] is True
    assert "Asha Verma" in out["tailoredResume"]
    # The prompt carried the pasted JD text in block J, not a sample fixture.
    assert "Backend engineer needed" in llm.prompts[0]
    # Approval + export re-validation pass for grounded content.
    svc.decide(out["draftId"], "approve")
    exported = svc.export_draft(out["draftId"])
    assert "Asha Verma" in exported["tailoredResume"]


def test_ai_target_draft_unknown_target_404s():
    svc = AISlateService(llm_client=MockLLM())
    rid = svc._upload_resume(RESUME)["resumeId"]
    with pytest.raises(SlateError) as e:
        svc.tailor_ai_target(rid, "target_nope")
    assert e.value.code == "target_not_found"


def test_ai_target_draft_ungrounded_lines_block_approval():
    # Model invents an employer that is in no source block and cites nothing.
    llm = MockLLM(reply="RESUME\nSenior engineer at Google, 5 years\nCOVER NOTE\nI fit [J].")
    svc = AISlateService(llm_client=llm)
    rid, tid = _seeded(svc)
    out = svc.tailor_ai_target(rid, tid)
    with pytest.raises(SlateError) as e:
        svc.decide(out["draftId"], "approve")
    assert e.value.code == "unsupported_claim"


def test_ai_target_draft_get_draft_rechecks_grounding():
    svc = AISlateService(llm_client=MockLLM(reply=GROUNDED_REPLY))
    rid, tid = _seeded(svc)
    out = svc.tailor_ai_target(rid, tid)
    fetched = svc.get_draft(out["draftId"])
    assert fetched["content"]["aiGenerated"] is True
    assert isinstance(fetched["violations"], list)


def test_ai_target_route_guard_and_flow(monkeypatch):
    app = create_authenticated_app(DevAuth(), MemUserStore(),
                                   llm_client=MockLLM(reply=GROUNDED_REPLY))
    c = TestClient(app)
    rid = c.post("/v1/resume", headers=H, json={"text": RESUME}).json()["resumeId"]
    tid = c.post("/v1/targets", headers=H,
                 json={"title": "Backend Engineer", "organization": "Zeta", "rawText": JD}
                 ).json()["target"]["targetId"]
    # Route refuses without a server key even though a client is injected.
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    off = c.post("/v1/tailor/ai/target", headers=H, json={"resumeId": rid, "targetId": tid})
    assert off.status_code == 503 and off.json()["error"]["code"] == "ai_not_configured"
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    on = c.post("/v1/tailor/ai/target", headers=H, json={"resumeId": rid, "targetId": tid})
    assert on.status_code == 200, on.text
    body = on.json()
    assert body["aiGenerated"] is True and body["requiresApproval"] is True
    did = body["draftId"]
    ok = c.post("/v1/approvals", headers=H, json={"draftId": did, "decision": "approve"})
    assert ok.status_code == 200, ok.text
    dl = c.get(f"/v1/drafts/{did}/download", headers=H)
    assert dl.status_code == 200 and "Asha Verma" in dl.text


def test_ai_target_draft_survives_restart(tmp_path, monkeypatch):
    # "Restart": a brand-new app over the same SQLite file still validates and
    # approves the target-based AI draft (grounding re-check finds the target).
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    url = f"sqlite:///{tmp_path}/state.db"
    c1 = TestClient(create_authenticated_app(DevAuth(), SqlUserStore(url),
                                               llm_client=MockLLM(reply=GROUNDED_REPLY)))
    rid = c1.post("/v1/resume", headers=H, json={"text": RESUME}).json()["resumeId"]
    tid = c1.post("/v1/targets", headers=H,
                  json={"title": "Backend Engineer", "organization": "Zeta", "rawText": JD}
                  ).json()["target"]["targetId"]
    did = c1.post("/v1/tailor/ai/target", headers=H,
                  json={"resumeId": rid, "targetId": tid}).json()["draftId"]
    c2 = TestClient(create_authenticated_app(DevAuth(), SqlUserStore(url),
                                               llm_client=MockLLM(reply=GROUNDED_REPLY)))
    fetched = c2.get(f"/v1/drafts/{did}", headers=H)
    assert fetched.status_code == 200 and fetched.json()["content"]["aiGenerated"] is True
    ok = c2.post("/v1/approvals", headers=H, json={"draftId": did, "decision": "approve"})
    assert ok.status_code == 200, ok.text
