"""The Gradio UI. Builds the 5 tabs and wires each button to a feature class."""

from __future__ import annotations

import gradio as gr
from google import genai

from config import Settings
from core import AnswerChecker, ExamPredictor, MaterialIngestor, PatternAnalyzer, QAEngine
from services import EmbeddingService, LLMService, VectorStore


def build_app(settings: Settings) -> gr.Blocks:
    client = genai.Client(api_key=settings.gemini_api_key)

    store = VectorStore(settings.vector_db_path, settings.collection_name)
    embedder = EmbeddingService(client, settings.embedding_model, settings.max_retries)
    llm = LLMService(client, settings.generation_model, settings.max_retries)

    ingestor = MaterialIngestor(store, embedder, settings.chunk_size, settings.chunk_overlap)
    pattern_analyzer = PatternAnalyzer(store, llm)
    qa_engine = QAEngine(store, embedder, llm, settings.top_k)
    answer_checker = AnswerChecker(store, embedder, llm)
    exam_predictor = ExamPredictor(store, llm)

    def handle_upload(files) -> str:
        if not files:
            return "⚠️ Please upload at least one file."
        try:
            result = ingestor.ingest([f.name for f in files])
        except Exception as exc:  # noqa: BLE001
            return f"❌ Something went wrong: {exc}"

        msg = f"✅ Indexed {result.files_processed} file(s), {result.chunks_indexed} chunk(s) total."
        if result.files_skipped:
            msg += f"\n⚠️ Skipped (unsupported or empty): {', '.join(result.files_skipped)}"
        return msg

    def handle_clear() -> str:
        store.clear()
        return "🗑️ All previously indexed materials have been cleared. Upload fresh files to start a new topic."

    def safe(fn, *args) -> str:
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            return f"❌ Something went wrong: {exc}"

    with gr.Blocks(title="GradEx AI — Exam Pattern Analyzer") as demo:
        gr.Markdown("# 📚 GradEx AI — Exam Pattern Analyzer & Study Assistant")
        gr.Markdown(
            "Upload your teacher's materials (PDF, PPT, DOCX, TXT) and let AI "
            "help you study smarter."
        )

        with gr.Tab("1️⃣ Upload Materials"):
            file_input = gr.File(file_count="multiple", label="Upload teacher's materials")
            upload_btn = gr.Button("Process Files", variant="primary")
            upload_output = gr.Textbox(label="Status", lines=3)
            upload_btn.click(handle_upload, inputs=file_input, outputs=upload_output)

            gr.Markdown("**Starting a new subject/topic?** Clear old materials first so answers don't mix old and new content.")
            clear_btn = gr.Button("🗑️ Clear All Indexed Materials", variant="secondary")
            clear_output = gr.Textbox(label="Clear status", lines=2)
            clear_btn.click(handle_clear, outputs=clear_output)

        with gr.Tab("2️⃣ Pattern Analysis"):
            analyze_btn = gr.Button("Analyze Teacher's Pattern", variant="primary")
            analyze_output = gr.Textbox(label="Analysis", lines=15)
            analyze_btn.click(lambda: safe(pattern_analyzer.analyze), outputs=analyze_output)

        with gr.Tab("3️⃣ Ask Questions"):
            question_input = gr.Textbox(label="Your question")
            ask_btn = gr.Button("Ask GradEx", variant="primary")
            ask_output = gr.Textbox(label="Answer", lines=10)
            ask_btn.click(lambda q: safe(qa_engine.answer, q), inputs=question_input, outputs=ask_output)

        with gr.Tab("4️⃣ Check My Answer"):
            check_question = gr.Textbox(label="Exam question")
            check_answer_box = gr.Textbox(label="Your answer", lines=5)
            check_btn = gr.Button("Check Answer", variant="primary")
            check_output = gr.Textbox(label="Feedback & Score", lines=12)
            check_btn.click(
                lambda q, a: safe(answer_checker.check, q, a),
                inputs=[check_question, check_answer_box],
                outputs=check_output,
            )

        with gr.Tab("5️⃣ Predict Exam Questions"):
            predict_btn = gr.Button("Predict Likely Questions", variant="primary")
            predict_output = gr.Textbox(label="Predictions", lines=15)
            predict_btn.click(lambda: safe(exam_predictor.predict), outputs=predict_output)

    return demo