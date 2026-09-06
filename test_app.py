"""Basic tests. Run with: pytest test_app.py"""

from extractors import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_short_text_returns_single_chunk():
    text = "hello world"
    assert chunk_text(text, chunk_size=100, overlap=10) == [text]


def test_long_text_splits_into_multiple_chunks():
    text = "a" * 250
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) == 4
    # consecutive chunks should share the overlap region
    assert chunks[0][-20:] == chunks[1][:20]
