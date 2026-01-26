# Purpose: Score and compare RAG pipeline outputs using LLM-as-judge
# Input: Question, generated answer, retrieved context (from each pipeline)
# Output: Scores for relevance, faithfulness, completeness + comparison

# =============================================================================
# WHAT THIS FILE DOES IN THE RAG PIPELINE:
# =============================================================================
# Multiple pipeline answers
#     ↓
# [evaluator.py] ← YOU ARE HERE
#     ↓
# Scores each answer: Pipeline A=8.5, Pipeline B=7.2, Pipeline C=9.1
#     ↓
# Returns comparison: "Pipeline C is 20% more accurate for your data"
# =============================================================================

# =============================================================================
# LLM-AS-JUDGE CONCEPT:
# =============================================================================
# Instead of manual human evaluation, we use a capable LLM to judge answers.
#
# Why it works:
#   - LLMs can assess relevance, coherence, factual grounding
#   - Scalable: Can evaluate thousands of answers automatically
#   - Consistent: Same criteria applied every time
#
# Important: Use a DIFFERENT (ideally stronger) model as judge than generator
#   - Generator: llama3.2 (3B) - fast, cheap
#   - Judge: llama3.1 (8B) or mistral - more capable
#
# This prevents the model from just agreeing with itself!
# =============================================================================

# =============================================================================
# EVALUATION METRICS FOR RAG:
# =============================================================================
# 1. RELEVANCE (Does the answer address the question?)
#    Score 1-10: How well does the answer match what was asked?
#    Low: "The sky is blue" for "How many PTO days?"
#    High: "You get 15 PTO days per year" for "How many PTO days?"
#
# 2. FAITHFULNESS (Is the answer grounded in the context?)
#    Score 1-10: Is the answer supported by the retrieved chunks?
#    Low: Answer includes facts NOT in the context (hallucination)
#    High: Every claim can be traced back to the context
#
# 3. COMPLETENESS (Did it capture all relevant info?)
#    Score 1-10: Does the answer include all relevant details from context?
#    Low: "You get PTO" (missing the number)
#    High: "You get 15 days PTO, with 5 days carryover allowed"
#
# 4. LATENCY (How fast was the response?)
#    Measured in milliseconds - comes from QueryResult, not LLM judge
#
# 5. COST (How many tokens used?)
#    Approximate token count for cost estimation
# =============================================================================

# =============================================================================
# JUDGE PROMPT TEMPLATE:
# =============================================================================
# The judge LLM needs clear instructions on how to score.
#
# JUDGE_PROMPT = """
# You are an expert evaluator for a question-answering system.
#
# Evaluate the following answer based on the question and context provided.
#
# Question: {question}
#
# Context (retrieved documents):
# {context}
#
# Generated Answer: {answer}
#
# Score the answer on these criteria (1-10 scale):
#
# 1. RELEVANCE: Does the answer directly address the question?
# 2. FAITHFULNESS: Is the answer fully supported by the context?
#    (penalize any information not found in context)
# 3. COMPLETENESS: Does the answer include all relevant information
#    from the context?
#
# Respond in JSON format:
# {
#     "relevance": <score>,
#     "faithfulness": <score>,
#     "completeness": <score>,
#     "reasoning": "<brief explanation>"
# }
# """
# =============================================================================

# =============================================================================
# CLASS DESIGN:
# =============================================================================
# @dataclass
# class EvaluationResult:
#     pipeline_name: str
#     question: str
#     answer: str
#     relevance: float       # 1-10
#     faithfulness: float    # 1-10
#     completeness: float    # 1-10
#     latency_ms: float
#     reasoning: str         # Judge's explanation
#
#     @property
#     def overall_score(self) -> float:
#         # Weighted average - faithfulness matters most for RAG
#         return (
#             self.relevance * 0.3 +
#             self.faithfulness * 0.4 +
#             self.completeness * 0.3
#         )
#
#
# class Evaluator:
#     def __init__(self, judge_model: str = "llama3.1:8b") -> None:
#         # Use a stronger model for judging
#         self.judge = Ollama(model=judge_model)
#
#     def evaluate_answer(
#         self,
#         question: str,
#         answer: str,
#         context: List[str],
#         pipeline_name: str,
#         latency_ms: float
#     ) -> EvaluationResult:
#         # Build judge prompt
#         # Call judge LLM
#         # Parse JSON response
#         # Return EvaluationResult
#
#     def compare_pipelines(
#         self,
#         results: List[EvaluationResult]
#     ) -> Dict[str, Any]:
#         # Aggregate scores across all questions
#         # Return comparison summary
#
#     def _parse_judge_response(self, response: str) -> Dict[str, float]:
#         # Extract JSON scores from judge output
# =============================================================================

# =============================================================================
# RUNNING EVALUATION (Full Flow):
# =============================================================================
# from services.rag_pipeline import RAGPipeline, PipelineConfig
# from services.evaluator import Evaluator, EvaluationResult
#
# # 1. Setup pipelines
# pipelines = [
#     RAGPipeline(config_small_nomic),
#     RAGPipeline(config_medium_nomic),
#     RAGPipeline(config_large_mxbai),
# ]
#
# # 2. Test questions
# test_questions = [
#     "How many PTO days do employees get?",
#     "What is the health insurance coverage?",
#     "How do I request time off?",
# ]
#
# # 3. Run all pipelines on all questions
# all_results = []
# for question in test_questions:
#     for pipeline in pipelines:
#         query_result = pipeline.query(question)  # Returns QueryResult
#
#         evaluation = evaluator.evaluate_answer(
#             question=question,
#             answer=query_result.answer,
#             context=query_result.retrieved_chunks,
#             pipeline_name=pipeline.config.name,
#             latency_ms=query_result.latency_ms
#         )
#         all_results.append(evaluation)
#
# # 4. Compare and summarize
# comparison = evaluator.compare_pipelines(all_results)
# print(comparison)
# # Output:
# # {
# #     "best_pipeline": "large_mxbai",
# #     "scores": {
# #         "small_nomic": {"avg_score": 7.2, "avg_latency": 85},
# #         "medium_nomic": {"avg_score": 7.8, "avg_latency": 120},
# #         "large_mxbai": {"avg_score": 8.5, "avg_latency": 180}
# #     },
# #     "recommendation": "large_mxbai scores 18% higher but is 2x slower"
# # }
# =============================================================================

# =============================================================================
# BATCH EVALUATION HELPER:
# =============================================================================
# For convenience, create a method that runs everything:
#
# class Evaluator:
#     ...
#
#     def run_benchmark(
#         self,
#         pipelines: List[RAGPipeline],
#         test_questions: List[str]
#     ) -> Dict[str, Any]:
#         """
#         Run all pipelines on all questions and return comparison.
#         """
#         results = []
#         for question in test_questions:
#             for pipeline in pipelines:
#                 result = pipeline.query(question)
#                 eval_result = self.evaluate_answer(...)
#                 results.append(eval_result)
#
#         return self.compare_pipelines(results)
# =============================================================================

# =============================================================================
# STORING RESULTS (for dashboard):
# =============================================================================
# Save evaluation results to SQLite for the frontend dashboard:
#
# import sqlite3
#
# def save_evaluation(result: EvaluationResult, db_path: str = "./results.db"):
#     conn = sqlite3.connect(db_path)
#     conn.execute("""
#         INSERT INTO evaluations
#         (pipeline_name, question, answer, relevance, faithfulness,
#          completeness, latency_ms, timestamp)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#     """, (
#         result.pipeline_name,
#         result.question,
#         result.answer,
#         result.relevance,
#         result.faithfulness,
#         result.completeness,
#         result.latency_ms,
#         datetime.now()
#     ))
#     conn.commit()
# =============================================================================

# =============================================================================
# HANDLING JUDGE FAILURES:
# =============================================================================
# The judge LLM might return malformed JSON or unexpected responses.
#
# def _parse_judge_response(self, response: str) -> Dict[str, float]:
#     try:
#         # Try to extract JSON from response
#         import json
#         import re
#
#         # Find JSON in response (might have extra text)
#         json_match = re.search(r'\{.*\}', response, re.DOTALL)
#         if json_match:
#             return json.loads(json_match.group())
#
#         raise ValueError("No JSON found in response")
#
#     except (json.JSONDecodeError, ValueError) as e:
#         # Fallback: return neutral scores
#         return {
#             "relevance": 5.0,
#             "faithfulness": 5.0,
#             "completeness": 5.0,
#             "reasoning": f"Failed to parse judge response: {e}"
#         }
# =============================================================================

# =============================================================================
# ALTERNATIVE: RAGAS FRAMEWORK
# =============================================================================
# RAGAS is a popular framework for RAG evaluation. If you want more
# sophisticated metrics, consider using it:
#
# pip install ragas
#
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision]
# )
#
# However, for learning purposes, building your own evaluator teaches
# you more about how RAG evaluation actually works.
# =============================================================================

# =============================================================================
# ERROR HANDLING TO CONSIDER:
# =============================================================================
# - Judge LLM not running / not installed
# - Malformed JSON response from judge
# - Empty context (no chunks retrieved)
# - Very long answers exceeding context window
# - Rate limiting on judge model
# =============================================================================

