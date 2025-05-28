import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Load audio file (replace 'your_audio.wav' with your file name)
with sr.AudioFile("your_audio.wav") as source:
    print("Listening...")
    audio = recognizer.record(source)

# Recognize speech using Google Web Speech API
try:
    text = recognizer.recognize_google(audio)
    print("\nTranscription:\n", text)
except sr.UnknownValueError:
    print("Could not understand audio.")
except sr.RequestError as e:
    print(f"Could not request results; {e}")
