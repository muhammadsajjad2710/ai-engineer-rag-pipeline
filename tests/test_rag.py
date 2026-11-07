from src.rag import Store, answer_with_citations
def test_retrieve_and_cite():
    s = Store()
    s.add("p1", "Returns are accepted within 30 days.")
    hits = s.search("return policy")
    ans = answer_with_citations(hits, "returns")
    assert ans["citations"] == ["p1"]
    assert "30 days" in ans["answer"]
