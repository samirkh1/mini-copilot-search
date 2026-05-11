import unittest

from mini_search.search_app import build_answer, first_sentence, retrieve, score_document, tokenize


class SearchAppTests(unittest.TestCase):
    def test_tokenize_removes_stopwords(self):
        tokens = tokenize("What is retrieval augmented generation?")
        self.assertIn("retrieval", tokens)
        self.assertIn("augmented", tokens)
        self.assertIn("generation", tokens)
        self.assertNotIn("what", tokens)
        self.assertNotIn("is", tokens)

    def test_score_document_counts_keyword_overlap(self):
        query = "RAG retrieval"
        document = "Retrieval augmented generation is also called RAG."
        self.assertEqual(score_document(query, document), 2)

    def test_retrieve_returns_best_matching_document(self):
        documents = {
            "docs/rag.txt": "RAG retrieves external evidence before generation.",
            "docs/git.txt": "Git uses branches and commits."
        }

        results = retrieve("How does RAG use evidence?", documents, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "docs/rag.txt")

    def test_build_answer_when_no_results(self):
        answer = build_answer("unknown query", [])

        self.assertIn("I don't have enough information", answer)
        self.assertIn("Sources:", answer)
        self.assertIn("None", answer)

    def test_build_answer_includes_sources(self):
        documents = {
            "docs/rag.txt": "RAG retrieves external evidence before generation."
        }

        results = retrieve("RAG evidence", documents)
        answer = build_answer("RAG evidence", results)

        self.assertIn("RAG retrieves external evidence", answer)
        self.assertIn("docs/rag.txt", answer)
    
    def test_first_sentence_returns_only_first_sentence(self):
        text = "First sentence. Second sentence."
        self.assertEqual(first_sentence(text), "First sentence.")


if __name__ == "__main__":
    unittest.main()
