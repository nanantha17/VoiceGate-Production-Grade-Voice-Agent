
import sounddevice as sd
import soundfile as sf

duration = 5
sample_rate = 16000
print("Recording 5 seconds... ask a question about the transformer paper")
audio = sd.rec(int(duration * sample_rate),
               samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
sf.write("user_question.wav", audio, sample_rate)
print("Saved to user_question.wav")