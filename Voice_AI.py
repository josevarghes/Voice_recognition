import speech_recognition as sr
import tkinter as tk
from tkinter import scrolledtext
from ttkthemes import ThemedTk
from tkinter import ttk

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration



recognizer = sr.Recognizer()
# Load the conversational model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Initialize conversation history
conversation_history = []

def capture_voice():
    with sr.Microphone() as source:
        print("Listening .....")
        audio = recognizer.listen(source)

    return audio

def  convert_voice_to_text(audio):
    try:
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text

    except :
        print("Something went wrong")

    
def get_chatbot_response(user_input):
    global conversation_history

    # Append user input to conversation history
    conversation_history.append(user_input)

    # Prepare the input for the model
    inputs = tokenizer(" ".join(conversation_history), return_tensors='pt')
    
    # Generate a response
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    # Append the response to the conversation history
    conversation_history.append(response)

    return response
   

def update_text(text):
    
     text_area.insert(tk.END, text)
     text_area.see(tk.END)



def main():
    input = capture_voice()
    voice_text = convert_voice_to_text(input)
    response = get_chatbot_response(voice_text)
    update_text(f"You: {voice_text}\nAI: {response}\n")


def start_listening():
    update_text("Listening...\n")
    main()

# Create the main window with a theme
root = ThemedTk(theme="breeze")
root.title("Voice Recognition AI Chatbot")

# Create a text area to display conversation with modern styling
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, font=("Arial", 12))
text_area.pack(padx=10, pady=10)

# Create a button to start listening with modern styling
listen_button = ttk.Button(root, text="Start Listening", command=start_listening)
listen_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()



