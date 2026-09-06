"""The 5 GradEx features. Each is a small class that combines the vector
store + Gemini services to do one job. Prompt templates are kept as
constants at the top so they're easy to find and tweak.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from extractors import chunk_text, extract_text
from services import EmbeddingService, LLMService, VectorStore

# ---------------------------------------------------------------- prompts --

PATTERN_ANALYSIS_PROMPT = """You are an AI that analyzes a teacher's pattern from their teaching materials.

Teacher's materials sample:
{context}

Based on these, identify:
1. Top 5 frequently discussed or important topics.
2. Important keywords the teacher uses.
3. What the teacher expects in answers (structure, examples, etc.).
4. Common mistakes that might lose marks.
5. Teacher's favorite examples or case studies.

Writing rules (follow these strictly):
- Use simple, everyday English. No jargon, no complex vocabulary, no long academic sentences.
- Under each numbered heading, write 3-5 short bullet points. One idea per bullet, one line each.
- Do NOT use bold text, sub-bullets, or nested formatting. Keep it flat and easy to scan.
- Explain any technical term in plain words the first time you use it.
- Be direct and specific — name the actual topic/number/example, don't describe it abstractly."""

QA_PROMPT = """You are GradEx, an AI assistant that answers questions using ONLY the
teacher's materials provided below.

Teacher's materials (relevant excerpts):
{context}

Question: {question}

Instructions:
- Use ONLY the materials above.
- If the answer is not clearly found, say so honestly.
- Include examples from the materials if available.
- Write in simple, everyday English — short sentences, no jargon, no unnecessary complexity.
- Keep the answer focused and well organized: a short direct answer first, then supporting
  points as plain bullets (no bold, no nested bullets)."""

ANSWER_CHECK_PROMPT = """You are a grading assistant. Compare the student's answer with the
teacher's materials and pattern.

Teacher's materials (relevant):
{context}

Question: {question}
Student's answer: {student_answer}

Analyze:
1. Did the student cover the key points expected by the teacher?
2. Are important keywords (from teacher's materials) used?
3. Is the answer structure appropriate?
4. Suggest a mark out of 10.
5. Provide specific improvement suggestions.

Writing rules: use simple, everyday English, short sentences, plain bullets under each
numbered point (no bold, no nested formatting). Be direct and specific."""

EXAM_PREDICTION_PROMPT = """You are an exam prediction expert. Based on the teacher's materials below,
predict likely exam questions.

Teacher's materials:
{context}

Provide:
1. TOP 5 topics that appear most frequently.
2. 3 topics that seem emphasized.
3. 2 difficult questions that might appear.
4. 1 tricky question students often get wrong.
5. Topics that can be safely skipped (rarely appear).

Writing rules: use simple, everyday English, short flat bullet points (no bold, no nested
formatting). Be specific and name the actual topics/questions, not vague descriptions."""


# ------------------------------------------------------------- ingestion --

@dataclass
class IngestionResult:
    files_processed: int
    files_skipped: list[str]
    chunks_indexed: int


class MaterialIngestor:
    """Pipeline: file -> extracted text -> chunks -> embeddings -> vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingService,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, filepaths: list[str | Path]) -> IngestionResult:
        skipped, total_chunks, processed = [], 0, 0

        for filepath in filepaths:
            path = Path(filepath)
            text = extract_text(path)
            if not text:
                skipped.append(path.name)
                continue

            chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
            if not chunks:
                skipped.append(path.name)
                continue

            embeddings = self._embedder.embed_documents_batch(chunks)
            total_chunks += self._store.add_chunks(chunks, embeddings, source=path.name)
            processed += 1

        return IngestionResult(processed, skipped, total_chunks)


# --------------------------------------------------------------- features --

class PatternAnalyzer:
    """Feature 2: summarize the teacher's recurring topics and expectations."""

    def __init__(self, store: VectorStore, llm: LLMService) -> None:
        self._store, self._llm = store, llm

    def analyze(self) -> str:
        if self._store.is_empty():
            return "⚠️ Please upload materials first (see Upload tab)."
        context = "\n\n---\n\n".join(self._store.sample(5))
        return self._llm.generate(PATTERN_ANALYSIS_PROMPT.format(context=context))


class QAEngine:
    """Feature 3: answer a question using only the retrieved teacher materials."""

    def __init__(self, store: VectorStore, embedder: EmbeddingService, llm: LLMService, top_k: int = 5) -> None:
        self._store, self._embedder, self._llm, self._top_k = store, embedder, llm, top_k

    def answer(self, question: str) -> str:
        if not question.strip():
            return "Please enter a question."
        if self._store.is_empty():
            return "⚠️ Please upload materials first (see Upload tab)."

        results = self._store.query(self._embedder.embed_query(question), self._top_k)
        if not results:
            return "No relevant information found in the teacher's materials."

        context = "\n\n---\n\n".join(r.text for r in results)
        response = self._llm.generate(QA_PROMPT.format(context=context, question=question))
        sources = sorted({r.source for r in results})
        return f"{response}\n\n📚 Sources: {', '.join(sources[:3])}"


class AnswerChecker:
    """Feature 4: grade a student's draft answer against the source material."""

    def __init__(self, store: VectorStore, embedder: EmbeddingService, llm: LLMService, top_k: int = 3) -> None:
        self._store, self._embedder, self._llm, self._top_k = store, embedder, llm, top_k

    def check(self, question: str, student_answer: str) -> str:
        if not question.strip() or not student_answer.strip():
            return "Please enter both a question and your answer."
        if self._store.is_empty():
            return "⚠️ Please upload materials first (see Upload tab)."

        results = self._store.query(self._embedder.embed_query(question), self._top_k)
        if not results:
            return "No teacher materials found relevant to this question."

        context = "\n".join(r.text for r in results)
        prompt = ANSWER_CHECK_PROMPT.format(
            context=context, question=question, student_answer=student_answer
        )
        return self._llm.generate(prompt)


class ExamPredictor:
    """Feature 5: predict likely exam questions from the full corpus."""

    def __init__(self, store: VectorStore, llm: LLMService, sample_size: int = 10) -> None:
        self._store, self._llm, self._sample_size = store, llm, sample_size

    def predict(self) -> str:
        if self._store.is_empty():
            return "⚠️ Please upload materials first (see Upload tab)."
        context = "\n".join(self._store.sample_random(self._sample_size))
        return self._llm.generate(EXAM_PREDICTION_PROMPT.format(context=context))