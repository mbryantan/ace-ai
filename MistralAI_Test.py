import threading
import os
import json
import subprocess
import webbrowser
import queue
import numpy as np
import sounddevice as sd
from mistralai import Mistral
from vosk import Model, KaldiRecognizer
import pyttsx3

api_key = "REDACTED"

client = Mistral(api_key=api_key)

mistral_model = "mistral-small-2508"

# whisper_model = whisper.load_model("base", device="cuda")
# if torch.cuda.is_available():
#     whisper_model = whisper_model.to(dtype=torch.float32)

# print("CUDA available:", torch.cuda.is_available())
# print("GPU name:", torch.cuda.get_device_name(0))

vosk_model_path = "vosk-model-en-us-0.22"
vosk_model = Model(vosk_model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)

engine = pyttsx3.init()
engine.setProperty("rate", 180)
engine.setProperty("volume", 1.0)

speech_queue = queue.Queue()

def tts_loop():
    """Continuously process queued text safely."""
    while True:
        text = speech_queue.get()  # Wait for next item
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start background thread for TTS
threading.Thread(target=tts_loop, daemon=True).start()

def speak(text):
    """Queue text for speaking (thread-safe)."""
    speech_queue.put(text)


sample_rate = 16000
buffer_secs = 5

q = queue.Queue()

print(sd.query_devices())

print("🤖 Mistral Assistant is ready! (type 'exit' to quit)\n")

# Conversation history (optional, to give context to replies)
messages = [{
    "role":"system",
    "content":"""You are a highly-advance futuristic AI, you act as my loyal assistant, and my wise advisor.
    Executing my commands, and answering to my questions. Reply to me as if we are having a conversation, and in this form

    Reply: <Your reply here>
    Action: {"action":"<action_name or none>"}

    if multiple actions requested, follow this format:
    Action: {"action": ["<action_name pr none>", "<action_name pr none>"]}

    Depending on how the phrase is delivered, here are the valid Actions, use this format to execute the necessary commands:
    - open_calculator
    - open_notepad
    - open_unity
    - open_google
    - open_youtube
    - open_messenger
    - none
    ...

    Never include extra fields, only "action".
"""
}]

# === Mistral Command Functions ===
def open_calculator():
    subprocess.Popen("calc.exe")

def open_notepad():
    subprocess.Popen("notepad")

def open_unity():
    subprocess.Popen("E:\\UnityHub\\Unity Hub\\Unity Hub.exe")

def open_website(url = "https://www.google.com/"):
    webbrowser.open(url)

# === Mic Functions ===
def audio_callback(indata, frames, time, status):
    """Callback function called for each audio block from mic."""
    if status:
        print(status)
    q.put(indata.copy())

def listen_and_transcribe():
    """Continuously listen and transcribe audio."""
    while True:
        audio_buffer = []

        # Collect audio for BUFFER_SECONDS
        for _ in range(int(sample_rate / 1024 * buffer_secs)):
            audio_buffer.append(q.get())

        audio_np = np.concatenate(audio_buffer, axis=0)
        # Flatten in case of stereo
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Check volume (basic silence detection)
        rms = np.sqrt(np.mean(audio_np**2))
        if rms < 0.0005:  # adjust threshold as needed
            continue  # skip transcription if too quiet

        # Convert float32 audio to int16 (vosk format)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        if recognizer.AcceptWaveform(audio_int16.tobytes()):
            result = json.loads(recognizer.Result())
            transcribed_text = result.get("text", "").strip()
        else:
            transcribed_text = ""

        if transcribed_text:
            print("🎤 You (voice):", transcribed_text)

            # Send exit signal
            if transcribed_text in ["exit", "quit", "stop listening"]:
                print("👋 Goodbye!")
                os.system(0)

            # Send to Mistral like typed input
            messages.append({"role": "user", "content": transcribed_text})
            response = client.chat.complete(model=mistral_model, messages=messages)

            assistant_reply = response.choices[0].message.content

            if "Action:" in assistant_reply:
                reply_text, action_part = assistant_reply.split("Action:", 1)
                reply_text = reply_text.strip().replace("Reply:", "").strip()
                action_json = action_part.strip()

                try:
                    start = action_json.find("{")
                    end = action_json.rfind("}") + 1
                    if start != -1 and end != -1:
                        action_json = action_json[start:end]
                    action = json.loads(action_json)
                    act = action.get("action")
                except Exception as e:
                    print("Error: ", e)

                print("Assistant:", reply_text)

                speak(reply_text)

                try:
                    action = json.loads(action_json)
                    act = action.get("action")

                    if isinstance(act, str) and act in ACTIONS:
                        threading.Thread(target=ACTIONS[act]).start()
                    elif isinstance(act, list):
                        for a in act:
                            if a in ACTIONS:
                                threading.Thread(target=ACTIONS[a]).start()
                except Exception as e:
                    print("⚠️ Failed to parse action:", e)
            else:
                print("Assistant:", assistant_reply)

                speak(assistant_reply)

            messages.append({"role": "assistant", "content": assistant_reply})

ACTIONS = {

    "open_calculator":open_calculator,
    "open_notepad":open_notepad,
    "open_unity": open_unity,
    "open_google": lambda: open_website(),
    "open_youtube": lambda: open_website("https://www.youtube.com/"),
    "open_messenger":  lambda: open_website("https://www.messenger.com/")

}

# Start the audio stream in a separate thread
stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=1024
)

stream.start()

# Start the transcription loop
threading.Thread(target=listen_and_transcribe, daemon=True).start()

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "stop listening"]:
        print("👋 Goodbye!")
        break

    # Add user message to conversation
    messages.append({"role": "user", "content": user_input})

    # Get response from Mistral
    response = client.chat.complete(model=mistral_model, messages=messages)

    # Extract reply
    assistant_reply = response.choices[0].message.content

    # Split reply and action
    if "Action:" in assistant_reply:
        reply_text, action_part = assistant_reply.split("Action:", 1)
        reply_text = reply_text.strip().replace("Reply:", "").strip()
        action_json = action_part.strip()

        try:
            start = action_json.find("{")
            end = action_json.rfind("}") + 1
            if start != -1 and end != -1:
                action_json = action_json[start:end]
            action = json.loads(action_json)
            act = action.get("action")
        except Exception as e:
            print("Error: ", e)

        # Print reply
        print("Assistant:", reply_text)

        speak(reply_text)

        # Try parsing action JSON
        try:
            action = json.loads(action_json)
            act = action.get("action")

            if isinstance(act, str) and act in ACTIONS:
                ACTIONS[act]()
            elif isinstance(act, list):
                for a in act:
                    if a in ACTIONS:
                        threading.Thread(target = ACTIONS[a]).start()

        except Exception as e:
            print("⚠️ Failed to parse action:", e)
    else:
        print("Assistant:", assistant_reply)

        speak(assistant_reply)

    # Add assistant reply to history
    messages.append({"role": "assistant", "content": assistant_reply})