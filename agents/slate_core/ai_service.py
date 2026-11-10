"""Opt-in AI tailoring on top of SlateService. Off by default.

The deterministic tailor path stays the default everywhere. This subclass adds
one method, tailor_ai, which asks an LLM (Groq by default) for a draft and then
keeps every safeguard the deterministic path has, plus more:

- the prompt is grounded: the model only sees numbered source blocks built
  deterministically from the uploaded resume and the job, and every content
  line must cite the blocks it uses
- grounding.check_grounding then mechanically re-checks those citations:
  uncited lines, unknown citations, numbers or capitalized names missing from
  the cited blocks, and lines whose content words mostly leave the cited
  blocks are all violations (see grounding.py for what it cannot catch)
- the draft enters the SAME approval store in "draft" status; approval and
  export re-validate through the normal _decide / export_draft path, which
  this class extends so grounding violations also block approval
- flagged skills and rule violations travel with the response so the review
  UI can show them before a human decides
- requiresApproval is always true; nothing auto-approves

Nothing here runs without LLM_API_KEY set on the server. The API layer also
refuses the route when the key is missing, so the feature is doubly off by
default.
"""
from . import jobs as J
from .claims import validate
from .grounding import build_grounded_prompt, check_grounding, source_blocks
from .llm import GeminiLLM, GroqLLM
from .service import SlateError, SlateService
from .tailor import check_claims


class AISlateService(SlateService):
    def __init__(self, store=None, llm_client=None):
        super().__init__(store)
        self._llm_client = llm_client  # tests inject MockLLM here
        # Interview coach provider, explicit: SLATE_INTERVIEW_PROVIDER=groq
        # (default, Slate's chosen provider, 8 Nov 2026) or gemini. A provider's
        # key is only ever sent to that provider's endpoint - never across.
        # With the default, a stray historical GEMINI_API_KEY does NOT reroute
        # interview data to Gemini; Gemini is used only when explicitly selected.
        # Missing the selected provider's key means the deterministic fallback,
        # never a silent provider swap.
        import os
        pref = (os.getenv("SLATE_INTERVIEW_PROVIDER") or "groq").strip().lower()
        coach_client = llm_client
        if coach_client is None and pref == "gemini":
            if os.getenv("GEMINI_API_KEY"):
                coach_client = GeminiLLM()
        elif coach_client is None:
            if os.getenv("LLM_API_KEY"):
                coach_client = GroqLLM()
        if coach_client is not None:
            from .interview import GeminiInterviewCoach
            self.interviews.coach = GeminiInterviewCoach(coach_client)

    def _job_for(self, job_id):
        """Sample-catalog job, or a pseudo-job built from a pasted target.

        Target pseudo-jobs keep sourceKind=user_pasted so nothing downstream
        can mistake pasted requirements for verified employer data."""
        job = J.get_job(job_id)
        if job is None and job_id in self._targets:
            t = self.get_target(job_id)
            job = {"jobId": job_id, "title": t["title"],
                   "company": t.get("company", ""), "location": "",
                   "skills": t.get("skills", []),
                   "description": t.get("rawText", ""),
                   "sourceKind": t.get("sourceKind", "user_pasted")}
        return job

    def _blocks(self, resume_id, job):
        return source_blocks(self._texts[resume_id], job)

    def tailor_ai(self, resume_id, job_id, client=None) -> dict:
        parsed = self._resume(resume_id)
        job = self._job_for(job_id)
        if job is None:
            raise SlateError("job_not_found", "No job or pasted target with that id.")
        client = client or self._llm_client or GroqLLM()
        blocks = self._blocks(resume_id, job)
        text = client.generate(build_grounded_prompt(blocks))
        resume, _, note = text.partition("COVER NOTE")
        resume = resume.replace("RESUME", "", 1).strip(" :\n")
        note = note.strip(" :\n")
        low = text.lower()
        claims = [s for s in job["skills"] if s.lower() in low]
        flagged = check_claims(parsed, claims)
        violations = validate(parsed, self._texts[resume_id], resume, note)
        violations += check_grounding(resume, blocks, allow_job=False)
        violations += check_grounding(note, blocks)
        did = self.approvals.create_draft(
            resume_id, job_id,
            {"tailoredResume": resume, "coverNote": note, "aiGenerated": True})
        self._persist()
        return {"draftId": did, "tailoredResume": resume,
                "coverNote": note, "status": "draft",
                "aiGenerated": True,
                "claimsChecked": claims, "skillGaps": [s for s in job["skills"] if s not in claims],
                "flaggedClaims": flagged, "violations": violations,
                "requiresApproval": True}

    def tailor_ai_target(self, resume_id, target_id, client=None) -> dict:
        """AI draft against a user-pasted job target. Same grounded prompt,
        citation re-checks and approval gate as tailor_ai; the draft binds to
        the target id, exactly like the deterministic tailor_target path."""
        if target_id not in self._targets:
            raise SlateError("target_not_found", "No pasted target with that id.")
        return self.tailor_ai(resume_id, target_id, client=client)

    def get_draft(self, draft_id):
        result = super().get_draft(draft_id)
        if result["content"].get("aiGenerated"):
            blocks = self._blocks(result["resumeId"], self._job_for(result["jobId"]))
            result["violations"] += check_grounding(result["content"].get("tailoredResume", ""), blocks, allow_job=False)
            result["violations"] += check_grounding(result["content"].get("coverNote", ""), blocks)
        return result

    def _check_draft(self, draft_id):
        """Approval re-checks the claim rules (super) AND the grounding of the
        stored content, so a draft with unsupported lines cannot be approved.
        Only drafts this class created (marked aiGenerated) get the grounding
        check; deterministic drafts keep the standard claim check only."""
        d = super()._check_draft(draft_id)
        if not d["content"].get("aiGenerated"):
            return d
        job = self._job_for(d.get("jobId"))
        blocks = self._blocks(d["resumeId"], job)
        bad = check_grounding(d["content"].get("tailoredResume", ""), blocks, allow_job=False)
        bad += check_grounding(d["content"].get("coverNote", ""), blocks)
        if bad:
            raise SlateError("unsupported_claim",
                             "Draft has lines not grounded in the resume: " + "; ".join(bad))
        return d
