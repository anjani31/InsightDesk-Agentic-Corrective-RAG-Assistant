"""RAGAS evaluation using local Qwen as the judge model."""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.config import settings
from app.graph import build_app
from app.ingestion import load_chunks

# Build an expert-reviewed golden set (expand to 50-200 in production):
GOLDEN_SET = [
    {"question": "How many leave days do new joiners get?",
     "ground_truth": "New joiners receive 18 days of annual leave, accrued monthly."},
    {"question": "What is the password reset procedure?",
     "ground_truth": "Users reset passwords via the self-service portal using MFA."},
]


def collect_predictions(agent):
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in GOLDEN_SET:
        result = agent.invoke(
            {"question": item["question"], "retry_count": 0},
            config={"configurable": {"thread_id": "eval"}},
        )
        rows["question"].append(item["question"])
        rows["answer"].append(result.get("generation", ""))
        rows["contexts"].append([d.page_content for d in result.get("documents", [])])
        rows["ground_truth"].append(item["ground_truth"])
    return Dataset.from_dict(rows)


def run_evaluation():
    agent = build_app(all_chunks=load_chunks("data"))
    dataset = collect_predictions(agent)
    judge_llm = ChatOllama(model=settings.LLM_MODEL,
                           base_url=settings.OLLAMA_BASE_URL, temperature=0)
    judge_emb = OllamaEmbeddings(model=settings.EMBEDDING_MODEL,
                                 base_url=settings.OLLAMA_BASE_URL)
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm, embeddings=judge_emb,
    )
    print(scores)
    return scores


if __name__ == "__main__":
    run_evaluation()
