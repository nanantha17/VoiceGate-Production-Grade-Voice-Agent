import os
import threading
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions
from dotenv import load_dotenv
from RAG_Chatbot import generate_answer


load_dotenv()
dg = DeepgramClient(os.getenv("DEEPGRAM_API_KEY", "DG_API_KEY"))


# ── STT: audio file → text (batch, nova-3) ─────────────────
def transcribe_audio(audio_file: str) -> str:
    try:
        with open(audio_file, "rb") as audio:
            response = dg.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio, "mimetype": "audio/wav"},
                PrerecordedOptions(
                    model="nova-3",  # latest model
                    smart_format=True,
                    punctuate=True,
                )
            )
        result = response.results.channels[0].alternatives[0]
        transcript = result.transcript
        confidence = result.confidence
        print(f"[STT] Transcript: {transcript}")
        print(f"[STT] Confidence: {confidence:.2f}")
        return transcript
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""


# ── TTS: text → audio (Aura) ───────────────────────────────
def synthesize_speech(text: str,
                      output_file: str = "response.mp3") -> str:
    try:
        dg.speak.rest.v("1").save(
            output_file,
            {"text": text},
            SpeakOptions(model="aura-asteria-en")
        )
        print(f"[TTS] Saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return ""


# ── Voice RAG pipeline ──────────────────────────────────────
def voice_rag(audio_file: "user_question.wav") -> str:
    print(f"\n[Pipeline] Starting voice RAG for: {audio_file}")

    # Step 1: STT
    user_text = transcribe_audio(audio_file)
    if not user_text:
        return "Could not transcribe audio"

    # Step 2: RAG
    rag_response = generate_answer(user_text)
    print(f"[RAG] Response: {rag_response}")

    # Step 3: TTS
    audio_out = synthesize_speech(rag_response)
    print(f"[Pipeline] Complete. Audio saved to: {audio_out}")

    return rag_response


# ── Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing TTS...")
    synthesize_speech(
        "Hello. THello. Deepgram TTS is working. "
        )
    print("Check response.mp3\n")