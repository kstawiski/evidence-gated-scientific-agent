from evals.knowledge_retrieval_benchmark import (
    _build_corpus,
    _descriptor_evidence_audit,
)


def test_benchmark_corpus_does_not_embed_semantic_relevance_ids():
    records = [
        {"id": "breast-dibh", "title": "Cardiac sparing", "text": "Evidence."},
        {"id": "lung-sbrt", "title": "Lung treatment", "text": "Evidence."},
    ]

    corpus, markers = _build_corpus(records)

    assert "breast-dibh" not in corpus
    assert "lung-sbrt" not in corpus
    assert markers == {
        "breast-dibh": "[SOURCE 001]",
        "lung-sbrt": "[SOURCE 002]",
    }


def test_descriptor_evidence_audit_checks_positive_hits_and_exact_source_slices():
    class Library:
        def search(self, query, snapshot, limit):
            del snapshot, limit
            if query == "valid":
                return {
                    "passages": [
                        {
                            "untrusted_source_text": "Exact immutable evidence.",
                            "retrieval_method": "semantic_descriptor",
                        }
                    ]
                }
            return {
                "passages": [
                    {
                        "untrusted_source_text": "Generated descriptor prose",
                        "descriptor": "must never be evidence",
                    }
                ]
            }

    violations, inspected = _descriptor_evidence_audit(
        Library(), object(), ["valid", "invalid"], "Exact immutable evidence."
    )

    assert inspected == 2
    assert violations == 1
