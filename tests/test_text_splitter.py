
import pytest

from src.text_splitter import split_text


def test_split_without_overlap():
    text = "ABCDEFGHIJ"

    chunks = split_text(text, chunk_size=4, overlap=0)

    assert chunks == ["ABCD", "EFGH", "IJ"]


def test_split_with_overlap():
    text = "ABCDEFGHIJ"

    chunks = split_text(text, chunk_size=4, overlap=1)

    assert chunks == ["ABCD", "DEFG", "GHIJ"]


def test_overlap_cannot_equal_chunk_size():
    with pytest.raises(ValueError):
        split_text("Hello World", chunk_size=4, overlap=4)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        split_text("Hello World", chunk_size=0, overlap=0)
