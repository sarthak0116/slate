"""Facade returning the contract shapes (contracts/api-contracts.md), errors in one shape."""
from . import jobs as J
from .resume import parse_resume, ParseError
from .tailor import build_draft
from .roadmap import build_roadmap
from .approvals import ApprovalStore
from .interview import InterviewStore
from .claims import validate

class SlateError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message

    def to_dict(self):
        return {"error": {"code": self.code, "message": self.message}}

class SlateService:
    def __init__(self, store=None):
        self._resumes = {}
        self._texts = {}
        self.approvals = ApprovalStore()
        self.interviews = InterviewStore()
        self._store = store
        if store is not None:
            st = store.load()
            if st:
                self._resumes, self._texts = st["resumes"], st["texts"]
                self.approvals._drafts, self.approvals._n = st["drafts"], st["draftN"]
                self.interviews._sessions, self.interviews._n = st["sessions"], st["sessionN"]

    def reset_demo(self, confirm: str) -> dict:
        """Demo helper, gated by the API layer: wipe resumes and drafts so the demo can restart.

        Interview sessions belong to the interview feature and are intentionally kept."""
        if confirm != "reset":
            raise SlateError("reset_not_confirmed", 'Send {"confirm": "reset"} to wipe the demo data.')
        self._resumes, self._texts = {}, {}
        self.approvals._drafts, self.approvals._n = {}, 0
        self._persist()
        return {"reset": True}

    def _persist(self):
        if self._store is not None:
            self._store.save({"resumes": self._resumes, "texts": self._texts,
                              "drafts": self.approvals._drafts, "draftN": self.approvals._n,
                              "sessions": self.interviews._sessions, "sessionN": self.interviews._n})

    def _resume(self, rid):
        r = self._resumes.get(rid)
        if r is None:
            raise SlateError("resume_not_found", "No resume with that id.")
        return r

    def _violations(self, resume_id, content) -> list:
        return validate(self._resume(resume_id), self._texts[resume_id],
                        content.get("tailoredResume", ""), content.get("coverNote", ""))

    def _check_draft(self, draft_id):
        d = self.approvals.get(draft_id)
        if d is None:
            raise SlateError("draft_not_found", "No draft with that id.")
        bad = self._violations(d["resumeId"], d["content"])
        if bad:
            raise SlateError("unsupported_claim", "Draft makes claims the resume does not support: " + "; ".join(bad))
        return d

    def _upload_resume(self, text) -> dict:
        try:
            parsed = parse_resume(text)
        except ParseError as e:
            raise SlateError("resume_unreadable", str(e))
        rid = f"res_{len(self._resumes) + 1:03d}"
        self._resumes[rid] = parsed
        self._texts[rid] = text
        return {"resumeId": rid, "parsed": parsed}

    def list_resumes(self):
        """Only this service's verified user's snapshots, newest first."""
        return {"totalCount": len(self._resumes), "resumes": [{"resumeId": rid, "name": parsed.get("name", "Unknown"),
                             "reviewStatus": parsed.get("skillReview", {}).get("status", "extracted_candidates"),
                             "parentResumeId": parsed.get("skillReview", {}).get("parentResumeId"),
                             "skillCount": len(parsed.get("skills", []))}
                            for rid, parsed in list(reversed(list(self._resumes.items())))[:100]]}

    def get_resume(self, resume_id):
        import copy
        parsed = self._resume(resume_id)
        original = parsed.get("skillReview", {}).get("originalResumeId", resume_id)
        return {"resumeId": resume_id, "parsed": copy.deepcopy(parsed),
                "sourceText": self._texts[original],
                "sourceTextKind": "original_uploaded_text_not_user_addendum"}

    def review_resume_skills(self, resume_id, skills, confirm=False):
        """A correction is a new immutable source snapshot, never silent reapproval."""
        from .resume_review import reviewed_snapshot
        parent = self._resume(resume_id)
        try:
            parsed = reviewed_snapshot(parent, skills, confirm)
        except ValueError as e:
            raise SlateError("invalid_skill_review", str(e))
        rid = f"res_{len(self._resumes) + 1:03d}"
        root_id = parent.get("skillReview", {}).get("originalResumeId", resume_id)
        original_text = self._texts[root_id]
        parsed["skillReview"] = {"parentResumeId": resume_id, "originalResumeId": root_id,
                               "status": "user_reviewed_not_independently_verified"}
        additions = [e["skill"] for e in parsed["skillEvidence"] if e["source"] == "user_statement"]
        # Additional claims are explicit user statements, not extracted PDF evidence.
        # Keep the original source snapshot intact. AI/draft gates still use the
        # new bound source, which labels the addendum and never implies work history.
        text = original_text + ("\n\nUser-provided skill statements (not independently verified):\nSkills: " + ", ".join(additions) if additions else "")
        self._resumes[rid], self._texts[rid] = parsed, text
        lineage, cursor = set(), resume_id
        while cursor and cursor not in lineage:
            lineage.add(cursor)
            cursor = self._resumes[cursor].get("skillReview", {}).get("parentResumeId")
        invalidated = []
        for did, draft in self.approvals._drafts.items():
            if draft["resumeId"] in lineage and draft["status"] == "approved":
                draft["status"] = "draft"
                invalidated.append(did)
        self._persist()
        return {"resumeId": rid, "parsed": parsed, "invalidatedDraftIds": invalidated}

    def jobs(self, query="", location="", resume_id=None) -> dict:
        skills = self._resume(resume_id)["skills"] if resume_id else []
        return {"jobs": J.search_jobs(query, location, skills)}

    def _tailor(self, resume_id, job_id) -> dict:
        parsed = self._resume(resume_id)
        job = J.get_job(job_id)
        if job is None:
            raise SlateError("job_not_found", "No job with that id.")
        d = build_draft(parsed, job)
        bad = self._violations(resume_id, d)
        if bad:  # generated drafts must pass the same check as edited ones
            raise SlateError("unsupported_claim", "Generated draft failed claim check: " + "; ".join(bad))
        did = self.approvals.create_draft(resume_id, job_id, {"tailoredResume": d["tailoredResume"], "coverNote": d["coverNote"]})
        return {"draftId": did, "tailoredResume": d["tailoredResume"], "coverNote": d["coverNote"],
                "status": "draft", "claimsChecked": d["claims"], "skillGaps": d["gaps"]}

    def _edit_draft(self, draft_id, **updates) -> dict:
        old = self.approvals.get(draft_id)
        if old is None:
            raise SlateError("draft_not_found", "No draft with that id.")
        bad = self._violations(old["resumeId"], {**old["content"], **updates})
        if bad:  # fail closed: no new draft is created
            raise SlateError("unsupported_claim", "Edit rejected, claims the resume does not support: " + "; ".join(bad))
        try:
            new_id = self.approvals.edit(draft_id, updates)
        except KeyError:
            raise SlateError("draft_not_found", "No draft with that id.")
        return {"draftId": new_id, "status": "draft"}

    def _decide(self, draft_id, decision) -> dict:
        if decision == "approve":
            self._check_draft(draft_id)  # re-validate at approval time
        try:
            status = self.approvals.decide(draft_id, decision)
        except KeyError:
            raise SlateError("draft_not_found", "No draft with that id.")
        except ValueError:
            raise SlateError("invalid_decision", "decision must be 'approve' or 'reject'.")
        except PermissionError:
            raise SlateError("draft_rejected", "A rejected draft cannot be reused. Create a new draft.")
        return {"draftId": draft_id, "status": status}

    def export_draft(self, draft_id) -> dict:
        try:
            content = self.approvals.export(draft_id)
            self._check_draft(draft_id)  # re-validate at export time
            return content
        except KeyError:
            raise SlateError("draft_not_found", "No draft with that id.")
        except PermissionError:
            raise SlateError("draft_not_approved", "Only an approved draft can be used outside the app.")

    def roadmap(self, resume_id, target_role) -> dict:
        ms = build_roadmap(self._resume(resume_id), target_role or "")
        if not ms:
            raise SlateError("role_unknown", "No sample listings for that role yet.")
        return {"milestones": ms}

    def _interview_resume_context(self, resume_id):
        if not resume_id:
            return {}
        parsed = self._resume(resume_id)
        # Send only bounded structured facts, never the original resume text.
        education = [
            {"degree": str(item.get("degree", ""))[:120],
             "institute": str(item.get("institute", ""))[:160],
             "year": item.get("year")}
            for item in parsed.get("education", []) if isinstance(item, dict)
        ][:12]
        experience = [
            {"title": str(item.get("title", ""))[:120],
             "org": str(item.get("org", ""))[:160],
             "months": item.get("months")}
            for item in parsed.get("experience", []) if isinstance(item, dict)
        ][:20]
        return {
            "name": str(parsed.get("name", ""))[:120],
            "skills": [str(skill)[:80] for skill in parsed.get("skills", [])[:80]],
            "education": education,
            "experience": experience,
        }
    def _start_interview(self, target_role, mode="text", resume_id=None) -> dict:
        if mode not in ("text", "voice"):
            raise SlateError("mode_unsupported", "Interview mode must be 'text' or 'voice'.")
        if not isinstance(target_role, str) or not target_role.strip():
            raise SlateError("role_unknown", "Enter a target role before starting the interview.")
        context = self._interview_resume_context(resume_id)
        return self.interviews.start(target_role.strip()[:120],
                                     self.interviews.past_feedback(target_role),
                                     resume_context=context, mode=mode)

    def _answer_interview(self, session_id, text, delivery_metrics=None) -> dict:
        try:
            return self.interviews.answer(session_id, text, delivery_metrics)
        except KeyError:
            raise SlateError("session_not_found", "No interview session with that id.")
        except ValueError as e:
            if str(e) == "answer_empty":
                raise SlateError("answer_empty", "Enter or dictate an answer before submitting.")
            if str(e) in ("interview_complete", "interview_ended"):
                raise SlateError("interview_ended", "This interview has ended.")
            raise

    def _end_interview(self, session_id) -> dict:
        try:
            return self.interviews.end(session_id)
        except KeyError:
            raise SlateError("session_not_found", "No interview session with that id.")

    def _interview_feedback(self, session_id) -> dict:
        try:
            return self.interviews.feedback(session_id)
        except KeyError:
            raise SlateError("session_not_found", "No interview session with that id.")

    def upload_resume(self, *a, **k):
        r = self._upload_resume(*a, **k)
        self._persist()
        return r

    def tailor(self, *a, **k):
        r = self._tailor(*a, **k)
        self._persist()
        return r

    def edit_draft(self, *a, **k):
        r = self._edit_draft(*a, **k)
        self._persist()
        return r

    def decide(self, *a, **k):
        r = self._decide(*a, **k)
        self._persist()
        return r

    def start_interview(self, *a, **k):
        r = self._start_interview(*a, **k)
        self._persist()
        return r

    def answer_interview(self, *a, **k):
        r = self._answer_interview(*a, **k)
        self._persist()
        return r

    def interview_feedback(self, *a, **k):
        r = self._interview_feedback(*a, **k)
        self._persist()
        return r

    def end_interview(self, *a, **k):
        r = self._end_interview(*a, **k)
        self._persist()
        return r
