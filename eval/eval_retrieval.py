"""
Retrieval + end-to-end evaluation harness for DocuChat.

Part 1 (retrieval-only, free, no API calls): compares naive vector search,
BM25 keyword search, hybrid ensemble, and cross-encoder reranking against a
hand-labeled question set built from the actual docs/ corpus (Google,
Microsoft, Nvidia, SpaceX, Tesla).

  Hit Rate@k = % of questions where the needed fact was present in top-k
  MRR@k      = mean reciprocal rank of the first chunk containing the fact

Part 2 (end-to-end, uses Azure OpenAI): for the baseline (naive vector) and
the best retrieval strategy, actually runs the full RAG pipeline (retrieve ->
prompt -> generate) and grades the generated answer against the same
ground-truth keywords. This is the number that matters for a resume claim
("the chatbot answered correctly"), not just "the right chunk was retrieved".

  Answer Accuracy = % of questions where the generated answer contains all
                     required fact keywords

Run:
    python eval_retrieval.py                  # retrieval-only
    python eval_retrieval.py --end-to-end      # also runs Part 2 (costs Azure OpenAI tokens)
"""

import os
import sys
import time

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

try:
    from langchain.retrievers import EnsembleRetriever  # langchain <1.0
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever  # langchain >=1.0
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

PERSIST_DIR = os.environ.get("EVAL_PERSIST_DIR", "data/chroma_db")
TOP_K = int(os.environ.get("EVAL_TOP_K", 5))

# ──────────────────────────────────────────────────────────────────
# Ground-truth question set (built from docs/*.txt content)
# Each question maps to the fact keywords that must be retrieved for
# an LLM to answer it correctly.
# ──────────────────────────────────────────────────────────────────
QUESTIONS = [
    # Google
    {"q": "Who founded Google and in what year?", "must_contain": ["Larry Page", "1998"]},
    {"q": "Who is the CEO of Google?", "must_contain": ["Sundar Pichai"]},
    {"q": "What is Google's parent company?", "must_contain": ["Alphabet"]},
    {"q": "What algorithm did Google's founders develop to rank search results?", "must_contain": ["PageRank"]},
    {"q": "What was Google's search engine originally nicknamed?", "must_contain": ["BackRub"]},
    {"q": "At which university were Larry Page and Sergey Brin PhD students?", "must_contain": ["Stanford"]},
    {"q": "What is Google's AI research lab called?", "must_contain": ["DeepMind"]},
    {"q": "What is Google's self-driving car project called?", "must_contain": ["Waymo"]},
    # Microsoft
    {"q": "Who founded Microsoft?", "must_contain": ["Bill Gates", "Paul Allen"]},
    {"q": "Where is Microsoft headquartered?", "must_contain": ["Redmond"]},
    {"q": "Who replaced Bill Gates as Microsoft CEO in 2000?", "must_contain": ["Steve Ballmer"]},
    {"q": "When did Satya Nadella become Microsoft's CEO?", "must_contain": ["2014"]},
    {"q": "How much did Microsoft pay to acquire LinkedIn?", "must_contain": ["26.2 billion"]},
    {"q": "How much did Microsoft pay to acquire Activision Blizzard?", "must_contain": ["68.7 billion"]},
    {"q": "What was the first company founded by Bill Gates and Paul Allen in 1972?", "must_contain": ["Traf-O-Data"]},
    {"q": "What was Microsoft's net income in 2025?", "must_contain": ["101.8 billion"]},
    # Nvidia
    {"q": "Who founded Nvidia and when?", "must_contain": ["Jensen Huang", "1993"]},
    {"q": "What share of the discrete GPU market does Nvidia hold?", "must_contain": ["92%"]},
    {"q": "What software platform did Nvidia develop for parallel computing on GPUs?", "must_contain": ["CUDA"]},
    {"q": "Where did Nvidia's three founders agree to start the company?", "must_contain": ["Denny's"]},
    {"q": "What was Nvidia's revenue in fiscal year 2025?", "must_contain": ["130.5 billion"]},
    {"q": "When did Nvidia surpass a $4 trillion market capitalization?", "must_contain": ["2025"]},
    {"q": "Who is Nvidia's chief scientist?", "must_contain": ["Bill Dally"]},
    {"q": "What mobile processor line did Nvidia develop?", "must_contain": ["Tegra"]},
    # SpaceX
    {"q": "Who founded SpaceX and when?", "must_contain": ["Elon Musk", "2002"]},
    {"q": "What was the first SpaceX rocket to successfully reach orbit?", "must_contain": ["Falcon 1"]},
    {"q": "When did SpaceX first land a Falcon 9 first stage?", "must_contain": ["2015"]},
    {"q": "What satellite internet constellation does SpaceX operate?", "must_contain": ["Starlink"]},
    {"q": "What is SpaceX's largest launch vehicle currently in development?", "must_contain": ["Starship"]},
    {"q": "Who is SpaceX's president and COO?", "must_contain": ["Gwynne Shotwell"]},
    {"q": "What percentage of SpaceX's voting control does Elon Musk hold?", "must_contain": ["79%"]},
    {"q": "What was SpaceX's estimated revenue in 2024?", "must_contain": ["10 billion"]},
    # Tesla
    {"q": "Who incorporated Tesla and when?", "must_contain": ["Eberhard", "2003"]},
    {"q": "What was Tesla's first production car model?", "must_contain": ["Roadster"]},
    {"q": "When did Tesla first become a trillion-dollar company?", "must_contain": ["2021"]},
    {"q": "What share of the battery electric vehicle market did Tesla have in 2024?", "must_contain": ["17.6%"]},
    {"q": "Who succeeded Eberhard as Tesla's CEO?", "must_contain": ["Musk"]},
    {"q": "How many vehicles did Tesla produce in 2024?", "must_contain": ["1,773,443"]},
    {"q": "What was Tesla's revenue in 2024?", "must_contain": ["97.7 billion"]},
    {"q": "Where is Tesla headquartered?", "must_contain": ["Austin"]},
]


def load_all_chunks(persist_directory: str):
    """Pull every chunk back out of the persisted Chroma store as Documents."""
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        collection_metadata={"hnsw:space": "cosine"},
    )
    raw = db.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return db, chunks


def build_retrievers(db, chunks, k=TOP_K):
    retrievers = {}

    retrievers["naive_vector"] = db.as_retriever(search_kwargs={"k": k})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    retrievers["bm25_keyword"] = bm25

    retrievers["hybrid_ensemble"] = EnsembleRetriever(
        retrievers=[bm25, db.as_retriever(search_kwargs={"k": k})],
        weights=[0.5, 0.5],
    )

    try:
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        class RerankRetriever:
            """Fetches a wide candidate pool from vector search, then reranks with a cross-encoder."""

            def __init__(self, base_retriever, encoder, top_k, fetch_k=20):
                self.base_retriever = base_retriever
                self.encoder = encoder
                self.top_k = top_k
                self.fetch_k = fetch_k

            def invoke(self, query):
                candidates = self.base_retriever.invoke(query)
                if not candidates:
                    return []
                pairs = [[query, doc.page_content] for doc in candidates]
                scores = self.encoder.predict(pairs)
                ranked = [doc for _, doc in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
                return ranked[: self.top_k]

        retrievers["reranked"] = RerankRetriever(
            db.as_retriever(search_kwargs={"k": 20}), cross_encoder, top_k=k
        )
    except Exception as e:
        print(f"[warn] Skipping cross-encoder reranker (not available: {e})")

    return retrievers


def contains_required_facts(text, must_contain):
    """True if every required keyword/phrase appears in text (case-insensitive)."""
    return all(kw.lower() in text.lower() for kw in must_contain)


def evaluate(retrievers, questions, k=TOP_K):
    results = {}
    for name, retriever in retrievers.items():
        hits = 0
        reciprocal_ranks = []
        for item in questions:
            docs = retriever.invoke(item["q"])[:k]
            rank_found = None
            for rank, doc in enumerate(docs, start=1):
                if contains_required_facts(doc.page_content, item["must_contain"]):
                    rank_found = rank
                    break
            if rank_found:
                hits += 1
                reciprocal_ranks.append(1.0 / rank_found)
            else:
                reciprocal_ranks.append(0.0)
        n = len(questions)
        results[name] = {
            "hit_rate": hits / n,
            "mrr": sum(reciprocal_ranks) / n,
            "hits": hits,
            "total": n,
        }
    return results


def build_llm():
    from langchain_openai import AzureChatOpenAI

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not api_key or not endpoint or not deployment_name:
        raise ValueError(
            "Azure OpenAI configuration is missing. Set AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME in .env to run --end-to-end."
        )

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment_name,
        temperature=0,
    )


def generate_answer(llm, retriever, question, k=TOP_K):
    docs = retriever.invoke(question)[:k]
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "Answer the question using ONLY the context below. "
        "Be specific and concise. If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)
    return response.content


def evaluate_end_to_end(llm, retrievers_subset, questions, k=TOP_K):
    results = {}
    for name, retriever in retrievers_subset.items():
        correct = 0
        transcripts = []
        for i, item in enumerate(questions, 1):
            answer = generate_answer(llm, retriever, item["q"], k=k)
            is_correct = contains_required_facts(answer, item["must_contain"])
            correct += is_correct
            transcripts.append({"q": item["q"], "answer": answer, "correct": is_correct})
            print(f"  [{name}] {i}/{len(questions)} {'OK' if is_correct else 'MISS'}: {item['q']}")
        n = len(questions)
        results[name] = {"accuracy": correct / n, "correct": correct, "total": n, "transcripts": transcripts}
    return results


def main():
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"{PERSIST_DIR} does not exist. Run `python learning/1_ingestion_pipeline.py` first "
            "(from the repo root) to build the vector store from data/sample_docs/."
        )

    print("Loading vector store and chunks...")
    db, chunks = load_all_chunks(PERSIST_DIR)
    print(f"Loaded {len(chunks)} chunks.\n")

    retrievers = build_retrievers(db, chunks, k=TOP_K)

    print(f"Evaluating {len(retrievers)} retrieval strategies on {len(QUESTIONS)} questions (k={TOP_K})...\n")
    results = evaluate(retrievers, QUESTIONS, k=TOP_K)

    print(f"{'Strategy':<20}{'Hit Rate@' + str(TOP_K):<15}{'MRR@' + str(TOP_K):<10}")
    print("-" * 45)
    for name, r in sorted(results.items(), key=lambda x: x[1]["hit_rate"], reverse=True):
        print(f"{name:<20}{r['hits']}/{r['total']} ({r['hit_rate']*100:.1f}%)".ljust(20 + 15) + f"{r['mrr']:.3f}")

    baseline = results.get("naive_vector")
    best_name, best = max(results.items(), key=lambda x: x[1]["hit_rate"])
    if baseline and best_name != "naive_vector":
        delta = (best["hit_rate"] - baseline["hit_rate"]) * 100
        print(f"\n'{best_name}' improved Hit Rate@{TOP_K} by {delta:+.1f} points over naive vector search "
              f"({baseline['hit_rate']*100:.1f}% -> {best['hit_rate']*100:.1f}%).")

    if "--end-to-end" in sys.argv:
        print("\n" + "=" * 60)
        print(f"PART 2: END-TO-END RAG ACCURACY (naive_vector vs {best_name})")
        print("=" * 60)
        llm = build_llm()
        subset = {"naive_vector": retrievers["naive_vector"], best_name: retrievers[best_name]}
        start = time.time()
        e2e_results = evaluate_end_to_end(llm, subset, QUESTIONS, k=TOP_K)
        elapsed = time.time() - start

        print(f"\n{'Strategy':<20}{'Answer Accuracy':<20}")
        print("-" * 40)
        for name, r in e2e_results.items():
            print(f"{name:<20}{r['correct']}/{r['total']} ({r['accuracy']*100:.1f}%)")

        base_acc = e2e_results["naive_vector"]["accuracy"] * 100
        best_acc = e2e_results[best_name]["accuracy"] * 100
        print(f"\nEnd-to-end eval took {elapsed:.1f}s for {len(QUESTIONS) * len(subset)} LLM calls.")
        print(f"'{best_name}' changed end-to-end answer accuracy by {best_acc - base_acc:+.1f} points "
              f"over naive vector search ({base_acc:.1f}% -> {best_acc:.1f}%).")

        misses = [t for t in e2e_results[best_name]["transcripts"] if not t["correct"]]
        if misses:
            print(f"\nQuestions '{best_name}' still got wrong:")
            for m in misses:
                print(f"  - {m['q']}\n    -> {m['answer'][:200]}")


if __name__ == "__main__":
    main()
