# chatbot_rag_eval.py
# Complete: Blenderbot + RAG (ChromaDB) + BLEU/ROUGE offline eval
# Run ingest_paper.py first to populate the vector DB

import chromadb
import json
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────
# 1. LOAD MODELS
# ─────────────────────────────────────────
print("Loading models...")

model_name = "google/flan-t5-base" #"facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────
# 2. CONNECT TO VECTOR DB
# ─────────────────────────────────────────
client = chromadb.PersistentClient(path="./attention_paper_db")
collection = client.get_or_create_collection("attention_paper")

MAX_CONTEXT_CHARS = 800 # was 400 — give FLAN-T5 more context to work with

# ─────────────────────────────────────────
# 3. RAG RETRIEVAL
# ─────────────────────────────────────────
def retrieve(query, n_results=5):  # get 5, rerank to top 3
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    chunks = results["documents"][0]
    distances = results["distances"][0]

    # Simple keyword reranking — boost chunks containing query keywords
    query_keywords = [w.lower() for w in query.split()
                      if len(w) > 3]  # skip short words

    def relevance_score(chunk):
        chunk_lower = chunk.lower()
        keyword_hits = sum(1 for kw in query_keywords
                           if kw in chunk_lower)
        return keyword_hits

    # Combine distance rank with keyword hits
    ranked = sorted(
        zip(chunks, distances),
        key=lambda x: (relevance_score(x[0]) * -1, x[1])  # more hits = better
    )

    top_chunks = [chunk for chunk, _ in ranked[:3]]
    return "\n\n".join(top_chunks)

def build_prompt(user_input, context, history):
    """
    Blenderbot has a 128-token input limit so we keep
    retrieved context tight and only use last 2 exchanges.
    """
    recent_history = " ".join(history[-4:])
    return (
        f"You are a precise question answering system. "
        f"Extract the exact answer from the context. "
        f"Include specific numbers, names, and technical terms.\n\n"
        f"Context: {context[:1200]}\n\n"
        f"Question: {user_input}\n\n"
        f"Answer:"
    )

def generate_answer(question, history=None):
    """Generate a grounded answer using RAG."""
    if history is None:
        history = []
    context = retrieve(question)
    prompt = build_prompt(question, context, history)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    outputs = model.generate(
        input_ids=inputs["input_ids"],  # pass explicitly, not **inputs
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=False,
        temperature=1.0,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ─────────────────────────────────────────
# 4. BLEU / ROUGE EVALUATION
# ─────────────────────────────────────────

# Reference Q&A pairs from "Attention Is All You Need"
# These are your ground truth answers for offline eval
EVAL_DATASET = [
    {
        "question": "What is the main contribution of the transformer?",
        "reference": "The transformer is the first transduction model relying entirely on self-attention to compute representations without using sequence-aligned RNNs or convolution."
    },
    {
        "question": "What is multi-head attention?",
        "reference": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions using h parallel attention heads."
    },
    {
        "question": "What are the three types of attention in the transformer?",
        "reference": "The transformer uses encoder-decoder attention, encoder self-attention, and decoder self-attention."
    },
    {
        "question": "What is the purpose of positional encoding?",
        "reference": "Positional encodings inject information about the relative or absolute position of tokens since the model contains no recurrence and no convolution."
    },
    {
        "question": "What is scaled dot-product attention?",
        "reference": "Scaled dot-product attention computes dot products of queries with all keys, divides by the square root of the key dimension, and applies softmax to get weights on the values."
    },
    {
        "question": "What BLEU score did the transformer achieve on English to German translation?",
        "reference": "The big transformer achieved 28.4 BLEU on WMT 2014 English to German translation, outperforming previously reported models including ensembles."
    },
    {
        "question": "Why does the transformer use self-attention instead of recurrence?",
        "reference": "Self-attention connects all positions with a constant number of operations while recurrent layers require O of n sequential operations, making self-attention faster when sequence length is smaller than representation dimensionality."
    },
    {
        "question": "What optimizer was used to train the transformer?",
        "reference": "The Adam optimizer was used with beta1 of 0.9, beta2 of 0.98, and epsilon of 10 to the negative 9 with a warmup learning rate schedule."
    },
]

def compute_bleu(reference, hypothesis):
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    smooth = SmoothingFunction().method1
    return {
        "bleu_1": round(sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smooth), 4),
        "bleu_2": round(sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth), 4),
        "bleu_4": round(sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth), 4),
    }

def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f1": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
    }

def run_eval():
    print("\n" + "=" * 85)
    print("OFFLINE EVALUATION — Attention Is All You Need RAG Chatbot")
    print("=" * 85)
    print(f"\n{'Question':<52} {'BLEU-1':>7} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}")
    print("-" * 85)

    results = []

    for item in EVAL_DATASET:
        question = item["question"]
        reference = item["reference"]
        hypothesis = generate_answer(question)

        bleu = compute_bleu(reference, hypothesis)
        rouge = compute_rouge(reference, hypothesis)

        results.append({
            "question": question,
            "reference": reference,
            "hypothesis": hypothesis,
            "bleu": bleu,
            "rouge": rouge,
        })

        q_short = question[:49] + "..." if len(question) > 49 else question
        print(
            f"{q_short:<52} "
            f"{bleu['bleu_1']:>7.4f} "
            f"{rouge['rouge1_f1']:>8.4f} "
            f"{rouge['rouge2_f1']:>8.4f} "
            f"{rouge['rougeL_f1']:>8.4f}"
        )

    # Aggregates
    avg = {
        "bleu_1":   sum(r["bleu"]["bleu_1"]    for r in results) / len(results),
        "bleu_4":   sum(r["bleu"]["bleu_4"]    for r in results) / len(results),
        "rouge_1":  sum(r["rouge"]["rouge1_f1"] for r in results) / len(results),
        "rouge_2":  sum(r["rouge"]["rouge2_f1"] for r in results) / len(results),
        "rouge_L":  sum(r["rouge"]["rougeL_f1"] for r in results) / len(results),
    }

    print("=" * 85)
    print(f"\nAverages:")
    print(f"  BLEU-1:  {avg['bleu_1']:.4f}  — unigram overlap")
    print(f"  BLEU-4:  {avg['bleu_4']:.4f}  — 4-gram overlap (strict; low is expected with Blenderbot)")
    print(f"  ROUGE-1: {avg['rouge_1']:.4f}  — unigram recall")
    print(f"  ROUGE-2: {avg['rouge_2']:.4f}  — bigram recall")
    print(f"  ROUGE-L: {avg['rouge_L']:.4f}  — primary metric (sequence-level recall + fluency)")

    # Flag low performers — retrieval quality check
    print(f"\nLow performers (ROUGE-L < 0.15) — investigate retrieval or chunking:")
    low = [r for r in results if r["rouge"]["rougeL_f1"] < 0.15]
    if low:
        for r in low:
            print(f"\n  Q:          {r['question']}")
            print(f"  Generated:  {r['hypothesis'][:100]}...")
            print(f"  Expected:   {r['reference'][:100]}...")
    else:
        print("  None — all questions above threshold.")

    # Save to JSON
    with open("eval_results.json", "w") as f:
        json.dump({"averages": avg, "per_question": results}, f, indent=2)
    print(f"\nFull results saved to eval_results.json")

# ─────────────────────────────────────────
# 5. CHATBOT LOOP
# ─────────────────────────────────────────
def chat_with_bot():
    history = []
    print("\nRAG Chatbot ready — ask questions about 'Attention Is All You Need'")
    print("Commands: 'eval' runs offline evaluation | 'quit' exits\n")

    while True:
        input_text = input("You: ").strip()

        if not input_text:
            continue

        # Run offline eval on demand
        if input_text.lower() == "eval":
            run_eval()
            continue

        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        history.append(input_text)

        response = generate_answer(input_text, history)
        history.append(response)

        # Show retrieved context so you can verify grounding
        context_preview = retrieve(input_text)[:150]
        print(f"\nChatbot: {response}")
        print(f"[Retrieved: {context_preview}...]\n")

# ─────────────────────────────────────────
# 6. DIAGNOSTICS LOOP
# ─────────────────────────────────────────

def debug_retrieval(question, top_k=3):
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    chunks = results["documents"][0]
    distances = results["distances"][0]  # lower = more similar in ChromaDB

    print(f"\nQuestion: {question}")
    print(f"{'=' * 60}")
    for i, (chunk, dist) in enumerate(zip(chunks, distances)):
        print(f"\nChunk {i + 1} (distance: {dist:.4f}):")
        print(chunk[:200])
        print(f"{'─' * 40}")

# ─────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    # ── run diagnostics first ──
    print("\nRunning retrieval diagnostics...")
    debug_retrieval("What is multi-head attention?")
    debug_retrieval("What optimizer was used to train the transformer?")
    debug_retrieval("What BLEU score did the transformer achieve?")
    print("\nDiagnostics complete. Starting chat...\n")
    #start chat

    chat_with_bot()