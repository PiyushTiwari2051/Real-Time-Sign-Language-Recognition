# utils/tts_utils.py
# This file provides an asynchronous wrapper for pyttsx3 text-to-speech.
# It uses a background thread and queue so speech does not freeze the webcam feed.

import queue
import threading
import pyttsx3

# Thread-safe queue for storing speech requests
speech_queue = queue.Queue()

def speech_worker():
    # Initialize the TTS engine within the background worker thread
    engine = pyttsx3.init()
    # Continuous loop waiting for texts to speak
    while True:
        # Get the next sentence or word from the queue (blocking wait)
        text = speech_queue.get()
        # Instruct the engine to speak
        engine.say(text)
        # Process speech output (blocking until speech finishes)
        engine.runAndWait()
        # Mark queue task as complete
        speech_queue.task_done()

# Start the background daemon thread to process speech requests
worker = threading.Thread(target=speech_worker, daemon=True)
worker.start()

def speak_text(text):
    # Queue up a text string to be spoken asynchronously
    speech_queue.put(text)

if __name__ == '__main__':
    # Print verification message and test speech output
    print("Testing tts_utils.py...")
    speak_text("Welcome to Sign Bridge Pro.")
    import time
    time.sleep(2)
    print("Speech test triggered successfully!")
