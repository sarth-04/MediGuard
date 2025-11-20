from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

class RagasEvaluator:
    def __init__(self, rag_pipeline):
        # Wrap the Gemini LLM and Embeddings so Ragas can use them
        self.eval_llm = LangchainLLMWrapper(rag_pipeline.llm)
        self.eval_embeddings = LangchainEmbeddingsWrapper(rag_pipeline.embeddings)

    def evaluate_response(self, query, response, context_list):
        """
        Runs Ragas evaluation on a single turn.
        Returns a dictionary with scores.
        """
        # Ragas expects a Dataset object
        data = {
            "question": [query],
            "answer": [response],
            "contexts": [context_list]
        }
        dataset = Dataset.from_dict(data)

        try:
            results = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=self.eval_llm,
                embeddings=self.eval_embeddings
            )
            
            # Convert result to a simple dict
            scores = results.to_pandas().iloc[0]
            return {
                "faithfulness": scores["faithfulness"],
                "answer_relevancy": scores["answer_relevancy"]
            }
        except Exception as e:
            print(f"Evaluation Error: {e}")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0}