"""
RAGAS Evaluation Script for the University Chatbot RAG pipeline.
Evaluates the RAG chain with 3 core RAGAS metrics:
  - Faithfulness:       Is the answer grounded in the retrieved context?
  - Answer Relevancy:   Does the answer address the question?
  - Context Precision:  Are the retrieved chunks relevant to the question?
"""

import os
import sys
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns


# Path setup — must be done before any local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from engine import process_pdf, get_embeddings, get_vectorstore, initialize_chatbot
from langchain_google_genai import ChatGoogleGenerativeAI

from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PDF_PATH = "data/txtleg-loi-n-01-00-Fr.pdf"

# Golden evaluation set (French — matches source documents)
QUESTIONS = [
    "Combien de semestres comporte un cycle de licence ?",
    "Le stage de recherche est-il obligatoire pour un Master ?",
    "Quelle est la note minimale pour valider un module ?",
    "Un étudiant peut-il refaire un semestre pour améliorer sa note ?",
    "Quel organe est responsable de la validation de l'organisation pédagogique ?",
]

GROUND_TRUTHS = [
    "Le cycle de licence est organisé en 6 semestres.",
    "Oui, un stage de recherche ou un mémoire est obligatoire au cours du dernier semestre du Master.",
    "Un étudiant doit obtenir une note d'au moins 10 sur 20 pour valider un module.",
    "Oui, un étudiant peut redoubler un semestre une seule fois pour tenter d'améliorer ses résultats.",
    "Le conseil de l'université est responsable de la validation de l'organisation pédagogique.",
]

# --------------------------------------------------
# Main evaluation
# --------------------------------------------------
def run_evaluation(pdf_path: str = PDF_PATH):
    print(f"\n{'='*60}")
    print("  RAGAS Evaluation — University Chatbot")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}\n")

    # 1. Process PDF & build vectorstore
    print("[1/4] Processing PDF and building vectorstore...")
    chunks = process_pdf(pdf_path)
    print(f"      → {len(chunks)} chunks created")

    with open(pdf_path, "rb") as f:
        pdf_hash = hashlib.md5(f.read()).hexdigest()
    vectorstore = get_vectorstore(chunks, pdf_hash)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. Initialize chatbot
    print("[2/4] Initializing chatbot...")
    bot = initialize_chatbot(chunks, pdf_hash)

    # 3. Generate answers + retrieve contexts
    print("[3/4] Running questions through the RAG pipeline...")
    answers, contexts = [], []

    for i, question in enumerate(QUESTIONS):
        print(f"      Q{i+1}: {question}")
        response = bot.invoke({"question": question})
        answers.append(response["answer"])
        docs = retriever.invoke(question)
        contexts.append([doc.page_content for doc in docs])

    # 4. Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question":     QUESTIONS,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": GROUND_TRUTHS,
    })

    # 5. Setup RAGAS judge (using Gemini 2.5 Flash for high performance)
    print("\n[4/4] Running RAGAS evaluation (this takes ~2 minutes)...")
    ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    ))

    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

    # Instantiate each metric and assign LLM/embeddings
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
    ]

    results = evaluate(dataset, metrics=metrics)

    # 6. Display results
    df = results.to_pandas()
    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS")
    print(f"{'='*60}")

    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    display_cols = [c for c in ["user_input", "faithfulness", "answer_relevancy", "context_precision"] if c in df.columns]

    # Fallback: show whatever columns exist
    if display_cols:
        print(df[display_cols].to_string(index=False))
    else:
        print(df.to_string(index=False))

    print(f"\n{'─'*40}")
    print("  AVERAGE SCORES")
    print(f"{'─'*40}")
    for col in ["faithfulness", "answer_relevancy", "context_precision"]:
        if col in df.columns:
            print(f"  {col.replace('_', ' ').title():25s}: {df[col].mean():.3f}  / 1.000")

    print(f"{'='*60}\n")

    # 7. Plot and Save Graph
    plot_results(df)


    # Save results
    output_path = "evals/ragas_results.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")

def plot_results(df):
    """Generate and save a bar chart of the average RAGAS metrics."""
    print("[5/5] Generating RAGAS metrics graph...")
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    averages = [df[m].mean() for m in metrics if m in df.columns]
    labels = [m.replace("_", " ").title() for m in metrics if m in df.columns]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Use a premium color palette
    colors = sns.color_palette("viridis", len(labels))
    
    bars = plt.bar(labels, averages, color=colors)
    
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0-1)")
    plt.title("RAGAS Evaluation Metrics — University Chatbot", fontsize=14, pad=20)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    graph_path = "evals/ragas_metrics_graph.png"
    plt.savefig(graph_path, dpi=300)
    print(f"✅ Graph saved to {graph_path}")


if __name__ == "__main__":

    run_evaluation()
