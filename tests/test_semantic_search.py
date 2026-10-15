import numpy as np
from src.semantic_search import calculate_similarity


def test_identical_vectors():
    documents = np.array([[1, 0]])
    question = np.array([1, 0])

    scores = calculate_similarity(documents, question)

    assert np.isclose(scores[0], 1.0)


def test_orthogonal_vectors():
    documents = np.array([[1, 0]])
    question = np.array([0, 1])

    scores = calculate_similarity(documents, question)

    assert np.isclose(scores[0], 0.0)