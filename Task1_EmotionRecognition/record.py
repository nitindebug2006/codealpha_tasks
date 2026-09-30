import sounddevice as sd
from scipy.io.wavfile import write

sample_rate = 22050
duration = 5

print("Recording started...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

write("recorded_audio.wav", sample_rate, audio)

print("Recording saved as recorded_audio.wav")