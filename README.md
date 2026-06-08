# RAG Conversational AI Agent
## Voice-Enabled Chatbot with Full Evaluation Pipeline

End-to-end RAG system with STT/TTS voice pipeline, ChromaDB vector retrieval,
and a complete offline-to-online evaluation stack.
Built to validate production readiness — not just functionality.

---

## Stack

| Layer | Technology |
|---|---|
| Speech-to-Text | Deepgram STT API |
| Text-to-Speech | Deepgram TTS API → response.mp3 |
| Embedding | HuggingFace sentence-transformers |
| Vector store | ChromaDB |
| LLM generation | RAG-augmented inference |
| Offline eval | BLEU / ROUGE + W&B LLM Evaluation |
| Online eval | CSAT, deflection rate, p50 latency |

---

## Pipeline Flow

```
User voice (wav)
    ↓
Deepgram STT → transcript text
    ↓
HuggingFace embedding → query vector
    ↓
ChromaDB → top-k relevant chunks
    ↓
LLM → RAG-augmented response text
    ↓
Deepgram TTS → response.mp3
    ↓
User hears answer
```

---

## File Structure

| File | Purpose |
|---|---|
| `record_question.py` | Captures user voice input → user_question.wav |
| `RAG_Chatbot.py` | Core RAG pipeline — embed, retrieve, generate |
| `RAG_chatbot_DG.py` | Deepgram STT/TTS integration layer |
| `user_question.wav` | Recorded voice input sample |
| `response.mp3` | Generated audio response sample |

---

## Evaluation Results

### Offline Evaluation (pre-deployment)

| Metric | Result |
|---|---|
| BLEU score | Tracked via W&B LLM Evaluation |
| ROUGE score | Tracked via W&B LLM Evaluation |
| Eval framework | W&B LLM Evals + custom eval harness |

Offline eval ran against a held-out QA set before any production traffic.
Pass criteria defined before eval ran — not tuned post-hoc.

### Online Evaluation (production)

| Metric | Result |
|---|---|
| CSAT | 82% |
| Deflection rate | 65% |
| p50 latency | 1.1s end-to-end |

**p50 of 1.1s** covers the full round trip: STT transcription +
embedding + ChromaDB retrieval + LLM inference + TTS synthesis.

---

## Key Design Decisions

**Why Deepgram for both STT and TTS?**
Single vendor for the voice layer reduces latency variance and simplifies
error handling — one auth flow, one SDK, consistent audio format handling
across input and output.

**Why ChromaDB?**
Persistent local vector store with no external API dependency for retrieval.
Retrieval latency stays in-process — keeps the p50 tight.

**Why offline eval before online?**
BLEU/ROUGE + W&B eval catches regression before users see it.
The offline → online transition is a deliberate gate, not an afterthought.
65% deflection rate validates the retrieval is surfacing relevant context —
not just generating fluent but wrong answers.

---

## Eval Pipeline as a Program Artifact

The eval pipeline is not a notebook — it is a program deliverable.

```
Offline gate:
  BLEU/ROUGE baseline established on v1
  W&B LLM Evals logged per model version
  Threshold defined: must exceed baseline before promotion

Online monitoring:
  CSAT collected per session
  Deflection tracked (did user ask a follow-up or accept the answer)
  p50 latency sampled per request

Version comparison:
  W&B run comparison between model versions
  Promotion blocked if CSAT drops >5% or p50 exceeds 1.5s
```

TPM implication: every model update triggers a re-run of the offline eval
before any production traffic. The online metrics are the acceptance
criteria, not the discovery mechanism.

---

## Running the Pipeline

```bash
# Install dependencies
pip install chromadb sentence-transformers deepgram-sdk wandb

# Record a question
python record_question.py

# Run RAG pipeline with voice output
python RAG_chatbot_DG.py

# Run offline eval
python RAG_Chatbot.py --eval
```

---

## Certifications

Built in conjunction with:
- W&B LLM Evaluation (2026)
- IBM GenAI & LLMs (Feb 2026)
- Building with Claude API — Anthropic (Apr 2026)
