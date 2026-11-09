import io
import re
from .skills import skill_evidence, resume_skill_evidence
from .resume_review import section_previews

MAX_PDF_BYTES = 5_000_000
MAX_RESUME_CHARS = 200_000
MAX_PDF_PAGES = 20

_QUALIFICATION_UNCERTAIN = re.compile(
    r"\b(?:no|not|never|without|lack|lacking|unfamiliar|seeking|aspir(?:e|es|ing)|"
    r"want(?:s)? to|wish(?:es)? to|plan(?:s|ning)? to|hop(?:e|es|ing) to|"
    r"interested in|learning|learn|instruction|example|pursuing|prospective)\b", re.I)

class ParseError(ValueError):
    pass

def parse_resume(text: str) -> dict:
    """Pull name, skills, education and experience from plain resume text."""
    if not isinstance(text, str) or len(text.strip()) < 20:
        raise ParseError("Resume text is empty or too short to parse.")
    if len(text) > MAX_RESUME_CHARS:
        raise ParseError("Resume text is too long (200,000 characters max).")
    text = _without_template_tips(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ParseError("No resume content remains after removing template guidance.")
    name = lines[0] if len(lines[0].split()) <= 5 and "@" not in lines[0] else "Unknown"
    evidence, uncertain = resume_skill_evidence(text)
    skills = [item["skill"] for item in evidence]
    education, experience = [], []
    for l in qualification_lines(text):
        m = re.search(r"(?<![a-z])(b\.?tech|m\.?tech|bca|mca|b\.?sc|m\.?sc|bachelor|master)(?![a-z])[^,\n]*", l, re.I)
        if m:
            y = re.search(r"(20\d\d)", l)
            education.append({"degree": m.group(1).upper().replace("..", "."), "institute": _after(l, ","),
                              "year": int(y.group(1)) if y else None})
        m = re.search(r"(intern|developer|analyst|engineer)\b.*?(?:at|,|-)\s*([A-Za-z0-9 &.]+?)\s*(?:\((\d+)\s*months?\)|$)", l, re.I)
        if m and not re.search(r"b\.?tech|bachelor", l, re.I):
            experience.append({"title": m.group(1).title(), "org": m.group(2).strip(),
                               "months": int(m.group(3)) if m.group(3) else None})
    return {"name": name, "skills": skills, "education": education, "experience": experience,
            "skillEvidence": evidence, "uncertainSkills": uncertain,
            "sectionReview": section_previews(text),
            "parseWarnings": ["Skill candidates from phrase matches and explicit lists, not verified proficiency or a complete resume assessment. "
                              "Review the detected skills; education and work-history extraction are limited."]}

def qualification_lines(text):
    """Conservative literal lines eligible for degree/work extraction, not proof."""
    excluded_section = False
    text = _without_template_tips(text)
    for l in text.splitlines():
        l = l.strip()
        label = l.casefold().rstrip(':：')
        if label.startswith("user-provided skill statements"):
            excluded_section = True
            continue
        # Explicit non-history sections can contain degree words and job titles
        # without describing the candidate's qualifications or employment.
        section_label = re.split(r"[:：]", label, maxsplit=1)[0].strip()
        if section_label in {"career objective", "objective", "career goals", "learning goals", "job requirements", "required qualifications", "example", "example experience", "instructions", "tips",
                             "skills", "key skills", "technical skills", "competencies", "core competencies", "compétences", "habilidades", "fähigkeiten", "कौशल", "技能",
                             "projects", "personal projects", "academic projects", "references", "referees", "interests", "hobbies", "interests/hobbies"}:
            excluded_section = True
            continue
        if label in {"education", "qualifications", "academic background", "experience", "work experience", "employment", "employment history", "professional experience", "formation", "educación", "ausbildung", "शिक्षा", "教育", "expérience", "experiencia", "berufserfahrung", "अनुभव", "工作经历"}:
            excluded_section = False
            continue
        # A literal degree/job-title mention is not necessarily a qualification.
        # Conservative exclusion keeps ambiguous text available for source review
        # rather than promoting a denial, aspiration or example into history.
        if excluded_section or _QUALIFICATION_UNCERTAIN.search(l):
            continue
        yield l


def find_skills(text: str) -> list:
    """Canonical skill mentions, including supported spelling/phrase aliases.

    Shared with claim checking. Do NOT strip template guidance here: a draft
    cannot hide a new skill from the claim gate by wrapping it in '(Tip: ...)'.
    """
    return [item["skill"] for item in skill_evidence(text)]


def _without_template_tips(text):
    """Ignore explicitly marked parenthesized resume-template instructions only.

    Handle line wraps and nested parentheses without swallowing later sections
    when an unclosed note occurs. Do not remove arbitrary parenthetical facts.
    """
    pattern = re.compile(r"\(\s*tip\s*:", re.I)
    pos, chunks = 0, []
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        depth, end = 1, match.end()
        while end < len(text) and depth:
            if text[end] == "(": depth += 1
            elif text[end] == ")": depth -= 1
            end += 1
        if depth:  # keep the unmatched remainder; never rescan it quadratically
            break
        chunks.append(text[pos:match.start()]); chunks.append("\n")
        pos = end
    chunks.append(text[pos:])
    return "".join(chunks)


def _after(line: str, sep: str) -> str:
    return line.split(sep, 1)[1].strip().rstrip(".") if sep in line else ""


def extract_pdf_text(data: bytes) -> str:
    """Pull selectable text out of PDF bytes. Scanned images and broken files raise ParseError."""
    if not data or len(data) > MAX_PDF_BYTES:
        raise ParseError("PDF is empty or larger than 5 MB.")
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ParseError("PDF support (pypdf) is not installed on this server.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ParseError("That PDF is password-protected; export it without a password and retry.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ParseError("PDF has too many pages (20 max). Export only your resume pages and retry.")
        chunks, total = [], 0
        for page in reader.pages:
            chunk = page.extract_text() or ""
            total += len(chunk) + 1
            if total > MAX_RESUME_CHARS:
                raise ParseError("PDF text is too long (200,000 characters max).")
            chunks.append(chunk)
        text = "\n".join(chunks)
    except ParseError:
        raise
    except Exception:
        raise ParseError("Could not read that file as a PDF.")
    if len(text.strip()) < 20:
        raise ParseError("No selectable text found in that PDF (a scanned image will not work).")
    return text
