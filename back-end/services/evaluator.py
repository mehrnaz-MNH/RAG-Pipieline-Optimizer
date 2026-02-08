
from dataclasses import dataclass
import json
from typing import Any, Dict, List
from langchain_community.llms import Ollama


@dataclass
class EvaluationResult:
    pipeline_name: str
    question: str
    answer: str
    relevance: float
    faithfulness: float
    completeness: float
    latency_ms: float
    reasoning: str

    @property
    def overall_score(self) -> float:
        # faithfulness matters most for RAG
        return (
            self.relevance * 0.3 +
            self.faithfulness * 0.4 +
            self.completeness * 0.3
        )


class Evaluator:
    def __init__(self, judge_model: str = "llama3.1:8b") -> None:

        self.judge = Ollama(model=judge_model)

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        context: List[str],
        pipeline_name: str,
        latency_ms: float
    ) -> EvaluationResult:
        JUDGE_PROMPT = """
        You are an expert evaluator for a question-answering system.
        Evaluate the following answer based on the question and context provided.

        Question: {question}

        Context (retrieved documents):
        {context}

        Generated Answer: {answer}

        Score the answer on these criteria (1-10 scale):

        1. RELEVANCE: Does the answer directly address the question?
        2. FAITHFULNESS: Is the answer fully supported by the context?
        (penalize any information not found in context)
        3. COMPLETENESS: Does the answer include all relevant information
        from the context?

        Respond in JSON format:
        {
          "relevance": <score>,
          "faithfulness": <score>,
          "completeness": <score>,
          "reasoning": "<brief explanation>"
        }
        """
        result = json.loads(self.judge.invoke(JUDGE_PROMPT))

        return EvaluationResult(
            pipeline_name = pipeline_name,
            question = question,
            answer = answer,
            relevance = result['relevance'],
            faithfulness = result['faithfulness'],
            completeness = result['completeness'],
            latency_ms = latency_ms,
            reasoning = result['reasoning'],
        )



    def compare_pipelines(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Returns the best pipeline name and its answer."""
        from collections import defaultdict

        # Group results by pipeline name
        by_pipeline = defaultdict(list)
        for r in results:
            by_pipeline[r.pipeline_name].append(r)

        # Calculate average score per pipeline
        avg_scores = {
            name: sum(r.overall_score for r in evals) / len(evals)
            for name, evals in by_pipeline.items()
        }

        # Find best pipeline
        best_name = max(avg_scores, key=avg_scores.get)

        # Get the best answer from the best pipeline
        best_result = max(by_pipeline[best_name], key=lambda r: r.overall_score)

        return {
            "best_pipeline": best_name,
            "answer": best_result.answer,
            "score": round(best_result.overall_score, 2)
        }


