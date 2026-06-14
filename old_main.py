import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import json
from google import genai
from google.genai import types
from pypdf import PdfReader
import re

app = FastAPI()

# This tells the server it is safe to talk to your web browser interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strict sequential schema for syllabus roadmap
class SubTopic(BaseModel):
    title: str
    description: str  # Quick summary of what to learn here
    estimated_minutes: int


class LearningStep(BaseModel):
    step_number: int  # e.g., 1, 2, 3 to enforce sequence
    step_title: str   # Main focus of this milestone
    difficulty: str   # Easy, Intermediate, Advanced
    sub_topics: List[SubTopic]  # Detailed breakdown in learning order


class StructuredSyllabus(BaseModel):
    course_name: str
    target_learning_flow: List[LearningStep]


class TopicStudyMaterialRequest(BaseModel):
    topic_title: str
    topic_description: str = ""
    raw_text: str = ""


class TopicStudyMaterialResponse(BaseModel):
    topic_title: str
    high_yield_summary: str
    must_know_definition: str
    common_student_trap: str
    active_recall_questions: List[str]
    active_recall_answers: List[str]
    practical_application: str


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9:,\- ]+", " ", text)).strip()


def _extract_ordered_topics(raw_text: str) -> List[str]:
    ordered: List[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(Module|Week|Unit|Chapter|Topic|Section)\s*[\dA-Za-z\-]*\s*[:\-]\s*(.+)$", line, re.I)
        if not match:
            continue
        payload = match.group(2)
        for part in re.split(r"[-,;/]", payload):
            topic = _clean_text(part).title()
            if len(topic) >= 4 and topic.lower() not in {"ltpc", "credits"}:
                ordered.append(topic)
    if ordered:
        return list(dict.fromkeys(ordered))

    for raw_line in raw_text.splitlines():
        line = _clean_text(raw_line).title()
        if 6 <= len(line) <= 70 and not re.search(r"\b(course code|instructor|credit hours|assessment)\b", line, re.I):
            ordered.append(line)
        if len(ordered) >= 8:
            break
    return list(dict.fromkeys(ordered))


def _chunk(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _difficulty_for(step_number: int, total_steps: int) -> str:
    if total_steps <= 2:
        return "Easy" if step_number == 1 else "Intermediate"
    if step_number == 1:
        return "Easy"
    if step_number == total_steps:
        return "Advanced"
    return "Intermediate"


API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if API_KEY:
    # Connect to the modern native Gemini AI client
    ai_client = genai.Client(api_key=API_KEY)
else:
    # No API key — run in local dev fallback mode (returns sample data)
    ai_client = None


@app.post("/api/parse-syllabus")
async def parse_syllabus(file: UploadFile = File(...)):
    raw_text = ""
    used_image_fallback = False
    content_type = file.content_type or ""
    filename = file.filename or "uploaded syllabus"

    if content_type == "application/pdf":
        try:
            pdf_reader = PdfReader(file.file)
            extracted_pages = [page.extract_text() or "" for page in pdf_reader.pages]
            raw_text = "\n".join(extracted_pages).strip()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to read PDF: {exc}") from exc

        if len(raw_text) < 50:
            raise HTTPException(
                status_code=400,
                detail="This PDF looks like a scanned image. Please upload a text-based PDF."
            )

    elif content_type.startswith("image/"):
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(file.file)
            raw_text = (pytesseract.image_to_string(image) or "").strip()
        except ImportError:
            used_image_fallback = True
            readable_name = re.sub(r"[_\-]+", " ", filename).strip()
            raw_text = (
                f"Uploaded syllabus image about {readable_name}. "
                "OCR libraries are unavailable, so inferred topic text from filename."
            )
        except Exception:
            used_image_fallback = True
            readable_name = re.sub(r"[_\-]+", " ", filename).strip()
            raw_text = (
                f"Uploaded syllabus image about {readable_name}. "
                "OCR extraction failed, so inferred topic text from filename."
            )

        if not used_image_fallback and len(raw_text) < 50:
            raise HTTPException(
                status_code=400,
                detail="Unable to extract readable text from this image. Please upload a clearer photo or a text-based PDF."
            )

    else:
        raise HTTPException(status_code=400, detail="File must be a PDF or image (PNG, JPG)")

    if ai_client is None:
        sample = generate_dev_roadmap(raw_text, filename)
        sample["raw_text"] = raw_text
        return sample

    prompt = f"""
You are an elite academic curriculum architect. Analyze this syllabus and reconstruct it into a flawless, logically ordered, step-by-step chronological learning pathway. Arrange the steps and sub-topics strictly in the exact order a student must master them, building from absolute foundational concepts up to complex applications. Ensure no topic jumps ahead of its prerequisites.

Syllabus Source Text:
{raw_text}
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuredSyllabus,
                temperature=0.1,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}") from exc

    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Backend returned invalid JSON") from exc

    parsed["raw_text"] = raw_text
    return parsed


@app.post("/api/topic-study-material", response_model=TopicStudyMaterialResponse)
async def generate_topic_study_material(payload: TopicStudyMaterialRequest):
    if ai_client is None:
        raise HTTPException(
            status_code=503,
            detail="AI study content generation is unavailable. Set GOOGLE_GENAI_API_KEY to enable Gemini-powered topic notes."
        )

    prompt = f"""
You are an elite university tutor.
Generate exam-focused study material for this topic.
Keep every section specific, practical, and directly useful for revision.

Topic:
{payload.topic_title}

Topic context:
{payload.topic_description}

Syllabus context:
{payload.raw_text}
"""

    schema = {
        "type": "object",
        "properties": {
            "topic_title": {"type": "string"},
            "high_yield_summary": {"type": "string"},
            "must_know_definition": {"type": "string"},
            "common_student_trap": {"type": "string"},
            "active_recall_questions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
            "active_recall_answers": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
            "practical_application": {"type": "string"},
        },
        "required": [
            "topic_title",
            "high_yield_summary",
            "must_know_definition",
            "common_student_trap",
            "active_recall_questions",
            "active_recall_answers",
            "practical_application",
        ],
    }

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}") from exc

    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Backend returned invalid JSON for topic study material") from exc

    try:
        return TopicStudyMaterialResponse(**parsed)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generated AI output did not match expected format: {exc}") from exc


def generate_dev_roadmap(raw_text: str, filename: str) -> dict:
    topics = _extract_ordered_topics(raw_text)
    if not topics:
        clean_name = filename.replace(".pdf", "").replace("_", " ").title()
        topics = [f"{clean_name} Foundations", "Core Concepts", "Applied Practice", "Advanced Integration"]

    grouped = _chunk(topics, 2)
    total_steps = len(grouped)
    flow = []
    for step_idx, group in enumerate(grouped, start=1):
        sub_topics = []
        for sub_idx, sub_title in enumerate(group, start=1):
            sub_topics.append(
                {
                    "title": sub_title,
                    "description": f"Master {sub_title} with definition, worked examples, and exam-style reasoning.",
                    "estimated_minutes": 35 + (sub_idx * 10),
                }
            )
        flow.append(
            {
                "step_number": step_idx,
                "step_title": f"Step {step_idx}: {group[0]}",
                "difficulty": _difficulty_for(step_idx, total_steps),
                "sub_topics": sub_topics,
            }
        )

    structured = {
        "course_name": f"{filename.replace('.pdf', '').replace('_', ' ').title()}",
        "target_learning_flow": flow,
    }
    return structured


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
