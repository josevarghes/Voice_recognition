import speech_recognition as sr
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from flask import Flask, request, jsonify, render_template

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Load the conversational model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Initialize conversation history
conversation_history = []

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_listening', methods=['POST'])
def start_listening():
    input_audio = capture_voice()
    voice_text = convert_voice_to_text(input_audio)
    if voice_text not in ["Sorry, I did not get that", "Could not request results; check your network connection", "Something went wrong"]:
        response = get_chatbot_response(voice_text)
        return jsonify({"user": voice_text, "bot": response})
    else:
        return jsonify({"error": voice_text})

def capture_voice():
    with sr.Microphone() as source:
        print("Listening .....")
        audio = recognizer.listen(source)
    return audio

def convert_voice_to_text(audio):
    try:
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not get that")
        return "Sorry, I did not get that"
    except sr.RequestError:
        print("Could not request results; check your network connection")
        return "Could not request results; check your network connection"
    except Exception as e:
        print(f"Something went wrong: {e}")
        return "Something went wrong"

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

if __name__ == '__main__':
    app.run(debug=True)
