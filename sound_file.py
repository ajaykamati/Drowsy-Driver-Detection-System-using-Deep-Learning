from playsound import playsound

from gtts import gTTS

text = "Don't sleep"
language = 'en'  # Language code (e.g., 'en' for English)

tts = gTTS(text=text, lang=language, slow=False)  # Create a gTTS object
tts.save("alarm.mp3")  # Save the audio to a file named "output.mp3"

alarm_sound_path = 'alarm.mp3'  # Replace with your audio file path
playsound(alarm_sound_path)