import re
import sys

from main import RAGPipeline
from eval.eval_set import EVAL_SET

class EvalRunner:

    def __init__(self):
        self.pipeline = RAGPipeline()

    @staticmethod
    def check_refusal_correctness(answer, expected_type):
        refused = "i can't determine" in answer.lower() or "contact" in answer.lower() and "tpa" in answer.lower()
        if expected_type == "refusal":
            return refused
        else:
            return not refused
    @staticmethod
    def check_keywords(answer, keyword_groups):
        """keyword_groups: list of lists — PASS if ANY group is fully present."""
        if not keyword_groups:
            return True
        return any(all(kw.lower() in answer.lower() for kw in group) for group in keyword_groups)

    @staticmethod
    def check_no_fabricated_number(answer, expected_type):
        if expected_type != "refusal":
            return True
        # a refusal answer should not also contain a confident rupee figure
        return not re.search(r'Rs\.?\s?[\d,]+', answer)

    def run_eval(self, eval_set):
        results = []
        for case in eval_set:
            result = self.pipeline.answer_question(case['query'], top_k=5, retrieve_k=15)
            answer = result['answer']
            row = {
                'query': case['query'],
                'expected_type': case['expected_type'],
                'refusal_correct': self.check_refusal_correctness(answer, case['expected_type']),
                'keywords_found': self.check_keywords(answer, case.get('must_contain_keywords', [])),
                'no_fabricated_number': self.check_no_fabricated_number(answer, case['expected_type']),
                'answer': answer,
                'top_chunk_score': result['retrieved_chunks'][0]['score'],
            }
            results.append(row)
        return results


if __name__ == "__main__":
    evalrunner = EvalRunner()
    results = evalrunner.run_eval(EVAL_SET)
    for r in results:
        passed = r['refusal_correct'] and r['keywords_found'] and r['no_fabricated_number']
        print(f"{'PASS' if passed else 'FAIL'} | {r['query']}")
        if not passed:
            print(f"   -> answer: {r['answer'][:200]}")