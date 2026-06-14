import json
import os
import re
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SubTopic(BaseModel):
    title: str
    description: str
    estimated_minutes: int


class LearningStep(BaseModel):
    step_number: int
    step_title: str
    difficulty: str
    sub_topics: List[SubTopic]


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


class TopicQuizRequest(BaseModel):
    topic_title: str
    topic_description: str = ""
    raw_text: str = ""
    difficulty: str = "Intermediate"
    ai_style: str = "Gemini"   # 👈 NEW: "ChatGPT", "Gemini", or "Claude"

class QuizQuestionItem(BaseModel):
    difficulty_level: str
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    ai_style: str  # ← ADD THIS (e.g., "ChatGPT", "Gemini", "Claude")


class TopicQuizResponse(BaseModel):
    topic_title: str
    quiz: List[QuizQuestionItem]


def _load_env_file() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _api_key() -> str | None:
    for name in ("GOOGLE_GENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


_load_env_file()
API_KEY = _api_key()
ai_client = None
if API_KEY:
    try:
        ai_client = genai.Client(api_key=API_KEY)
    except Exception as exc:
        print(f"WARNING: Gemini client failed to initialize: {exc}")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ai_enabled": ai_client is not None,
        "ai_mode": "gemini" if ai_client is not None else "local_fallback",
    }


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9:,\- ]+", " ", text)).strip()


def _extract_ordered_topics(raw_text: str) -> List[str]:
    ordered: List[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(
            r"^(Module|Week|Unit|Chapter|Topic|Section)\s*[\dA-Za-z\-]*\s*[:\-]\s*(.+)$",
            line,
            re.I,
        )
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
        if 6 <= len(line) <= 70 and not re.search(
            r"\b(course code|instructor|credit hours|assessment)\b", line, re.I
        ):
            ordered.append(line)
        if len(ordered) >= 8:
            break
    return list(dict.fromkeys(ordered))


def _chunk(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _difficulty_for(step_number: int, total_steps: int) -> str:
    if total_steps <= 2:
        return "Easy" if step_number == 1 else "Intermediate"
    if step_number == 1:
        return "Easy"
    if step_number == total_steps:
        return "Advanced"
    return "Intermediate"


def generate_dev_roadmap(raw_text: str, filename: str) -> dict:
    topics = _extract_ordered_topics(raw_text)
    if not topics:
        clean_name = filename.replace(".pdf", "").replace("_", " ").title()
        topics = [
            f"{clean_name} Foundations",
            "Core Concepts",
            "Applied Practice",
            "Advanced Integration",
        ]

    grouped = _chunk(topics, 2)
    total_steps = len(grouped)
    flow = []
    for step_idx, group in enumerate(grouped, start=1):
        sub_topics = []
        for sub_idx, sub_title in enumerate(group, start=1):
            sub_topics.append(
                {
                    "title": sub_title,
                    "description": (
                        f"Master {sub_title} with definition, worked examples, "
                        "and exam-style reasoning."
                    ),
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

    return {
        "course_name": filename.replace(".pdf", "").replace("_", " ").title(),
        "target_learning_flow": flow,
    }


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
                detail="This PDF looks like a scanned image. Please upload a text-based PDF.",
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
                detail=(
                    "Unable to extract readable text from this image. "
                    "Please upload a clearer photo or a text-based PDF."
                ),
            )

    else:
        raise HTTPException(status_code=400, detail="File must be a PDF or image (PNG, JPG)")

    if ai_client is None:
        sample = generate_dev_roadmap(raw_text, filename)
        sample["raw_text"] = raw_text
        return sample

    prompt = f"""
You are an elite academic curriculum architect. Analyze this syllabus and reconstruct it into a flawless, logically ordered, step-by-step chronological learning pathway.

Requirements:
- Set `course_name` from the syllabus title or subject.
- Use `target_learning_flow` with 4–8 learning steps when the syllabus is substantial (fewer if the syllabus is short).
- Each step must have `step_number`, `step_title`, `difficulty` (Easy, Intermediate, or Advanced), and `sub_topics`.
- Each sub-topic needs a specific `title`, a practical one-sentence `description`, and realistic `estimated_minutes` (30–120).
- Order topics so prerequisites always come first; never skip ahead to advanced material.
- Extract real module/week/topic names from the syllabus text; do not invent generic placeholders.

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
            detail=(
                "AI study content generation is unavailable. "
                "Set GOOGLE_GENAI_API_KEY in backend/.env to enable Gemini-powered topic notes."
            ),
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
            "active_recall_questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
            "active_recall_answers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
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
        raise HTTPException(
            status_code=502,
            detail="Backend returned invalid JSON for topic study material",
        ) from exc

    try:
        return TopicStudyMaterialResponse(**parsed)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Generated AI output did not match expected format: {exc}",
        ) from exc


def _fallback_topic_quiz(
    topic_title: str,
    topic_description: str,
    difficulty: str,
) -> dict:
    """Deterministic quiz when Gemini is unavailable."""
    """Deterministic quiz when Gemini is unavailable."""
    
    title = topic_title.strip() or "this topic"
    desc = (topic_description or title).strip()
    return {
        "ai_style": "ChatGPT",
        "topic_title": title,
        "quiz": [
            {
                "difficulty_level": "Easy",
                "question": f"What is the central learning objective of {title}?",
                "options": [
                    f"Apply core concepts of {title} to explain and solve course problems",
                    "Memorize unrelated facts with no link to the syllabus",
                    "Skip foundational ideas and only read the unit title",
                    "Avoid practice because this topic is never assessed",
                ],
                "correct_answer": f"Apply core concepts of {title} to explain and solve course problems",
                "explanation": f"Syllabus focus: {desc}",
            },
            {
                "difficulty_level": "Moderate",
                "question": f"Which strategy best prepares you for an exam question on {title}?",
                "options": [
                    "State definitions, study a worked example, then solve a similar problem",
                    "Only re-read the heading without notes or practice",
                    "Study unrelated units instead of this topic",
                    "Copy solutions without understanding the method",
                ],
                "correct_answer": "State definitions, study a worked example, then solve a similar problem",
                "explanation": "Exam-style mastery requires understanding plus deliberate practice.",
            },
            {
                "difficulty_level": "Moderate",
                "question": f"A scenario-based question mentions {title}. What should you do first?",
                "options": [
                    "Identify knowns/unknowns, recall the relevant principle, then reason step by step",
                    "Guess the letter immediately with no written work",
                    "Leave it blank because it was not in the title",
                    "Use a rule from a completely different module",
                ],
                "correct_answer": "Identify knowns/unknowns, recall the relevant principle, then reason step by step",
                "explanation": f"Structured reasoning is expected at {difficulty} level.",
            },
            {
                "difficulty_level": "Tough",
                "question": f"Which misconception about {title} is most dangerous before the final?",
                "options": [
                    "Assuming you can skip it because later topics never build on it",
                    "Checking whether your answer matches units and logical constraints",
                    "Relating new problems to a previously solved example",
                    "Reviewing mistakes from practice questions",
                ],
                "correct_answer": "Assuming you can skip it because later topics never build on it",
                "explanation": "Syllabus order usually encodes prerequisites.",
            },
            {
                "difficulty_level": "Tough",
                "question": f"How would an expert instructor evaluate your understanding of {title}?",
                "options": [
                    "You can explain the idea, justify each step, and transfer it to a novel scenario",
                    "You can spell the topic name correctly only",
                    "You list chapter numbers without explaining mechanisms",
                    "You recall one memorized answer but cannot adapt it",
                ],
                "correct_answer": "You can explain the idea, justify each step, and transfer it to a novel scenario",
                "explanation": "Deep understanding shows in explanation and transfer, not recall alone.",
            },
        ],
    }
def _get_style_instruction(ai_style: str) -> str:
    styles = {
        "ChatGPT": """
You write questions like ChatGPT — friendly, practical, and real-world focused.
- Use everyday relatable scenarios (e.g., "You are building an app that...")
- Questions feel like a helpful tutor is asking them
- Options are clear, not tricky
- Easy to understand even for beginners
""",
        "Gemini": """
You write questions like Gemini — structured, multi-step, and concept-focused.
- Ask "which of the following best explains..." type questions
- Include questions that require connecting two ideas together
- Options are detailed and test deeper understanding
- Professional and textbook-like tone
""",
        "Claude": """
You write questions like Claude — thoughtful, nuanced, and edge-case focused.
- Include "what is wrong with this approach?" style questions
- Test misconceptions and subtle differences between similar concepts
- At least one question should present a flawed code/logic and ask what's wrong
- Encourage critical thinking over memorization
""",
    }
    # If unknown style, default to Gemini style
    return styles.get(ai_style, styles["Gemini"])

@app.post("/api/topic-quiz", response_model=TopicQuizResponse)
async def generate_topic_quiz(payload: TopicQuizRequest):
    title = payload.topic_title.strip() or "Topic"
    description = payload.topic_description.strip()
    difficulty = payload.difficulty.strip() or "Intermediate"
    context = (payload.raw_text or "")[:12000]

    if ai_client is None:
        return TopicQuizResponse(**_fallback_topic_quiz(title, description, difficulty))
    style_instruction = _get_style_instruction(payload.ai_style)
    prompt = f"""

You are an expert university instructor creating a multiple-choice practice quiz.
{style_instruction}

Topic: {title}
Topic summary: {description or "See syllabus context."}
Course difficulty band: {difficulty}

Syllabus context (use for accuracy; do not invent unrelated content):
{context or "No extra syllabus text provided — stay faithful to the topic title and summary."}

Requirements:
- Return exactly 5 questions.
- Each question must have exactly 4 distinct answer options (full sentences, not single letters).
- Mix difficulty_level values: include at least one Easy, two Moderate, and two Tough.
- Questions must test understanding, application, and common misconceptions — not meta questions about studying.
- correct_answer must match one option exactly (character-for-character).
- explanation: 1–2 sentences clarifying why the correct option is right.
- Avoid trick questions, avoid "all of the above", avoid duplicate options.

"""

    schema = {
        "type": "object",
        "properties": {
            "topic_title": {"type": "string"},
            "quiz": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    
                    "type": "object",
                    "properties": {
                        "ai_style": {
                             "type": "string",
                             "enum": ["ChatGPT", "Gemini", "Claude", "Perplexity", "Mixed"],
                             },
                        "difficulty_level": {
                            "type": "string",
                            "enum": ["Easy", "Moderate", "Tough"],
                        },
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "correct_answer": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "difficulty_level",
                        "question",
                        "options",
                        "correct_answer",
                        "explanation",
                    ],
                },
            },
        },
        "required": ["topic_title", "quiz"],
    }

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.35,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI quiz generation failed: {exc}") from exc

    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Backend returned invalid JSON for quiz") from exc

    try:
        return TopicQuizResponse(**parsed)
    except Exception:
        return TopicQuizResponse(**_fallback_topic_quiz(title, description, difficulty))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
