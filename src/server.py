from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from chatbot import get_chat_pipeline
from chatcontext import ChatContext, PartialMessage, ChatState
from ulid import ULID
from acronym import load_acronyms
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)
app.config["DEBUG"] = False
CORS(app, resources={r"/api/*": {"origins": "*"}})

language = "fr"
chat_pipeline = get_chat_pipeline(language)

chat_contexes: dict[str, ChatContext] = dict()
try:
    with open("./discussions.json") as file:
        raw_contexes = json.load(file)
        assert type(raw_contexes) == dict
        chat_contexes = {key: ChatContext.deserialize(value) for key, value in raw_contexes.items()}
except Exception as e:
    print(e)
    print("Discussions could not be loaded, skipping!")
    pass

presentation = {
    "en": "Hello, I am Bob, your medical chatbot. Feel free to ask me any question about your heart!",
    "fr": "Bonjour, je suis Bob, votre chatbot médical. N'hésitez pas à me demander des questions à propos de votre coeur!"
}[language]

def save_conversations():
    serialized = {key: context.serialize() for key, context in chat_contexes.items()}
    with open("./discussions.json", "w") as file:
        json.dump(serialized, file)

@app.route("/api/chat", methods=["POST"])
@cross_origin()
def create_chat():
    chat = ChatContext()
    chat_contexes[chat.id] = chat

    # TODO: remove, this is for testing purposes only
    message = PartialMessage()
    message.set_content(presentation, "presentation")
    chat.append_message(message.build())

    return jsonify(chat.serialize())

@app.route("/api/chat/<chat_id>", methods=["POST"])
def post_message(chat_id: str):
    data = request.get_json()
    if not (chat_id in chat_contexes):
        raise ValueError(f"Chat {chat_id} not found!")
    if not ("message" in data) or type(data["message"]) != str:
        raise ValueError(f"Key 'message' not in body or not a string!")

    chat: ChatContext = chat_contexes[chat_id]

    if chat.state != ChatState.Idle:
        raise ValueError(f"Chat {chat_id} is already answering!")

    user_message = PartialMessage()
    user_message.set_content(data["message"], "user")
    user_message.set_metadata("user", True)

    chat_pipeline.get_answer(chat, user_message.build())

    # chat.append_message(user_message.build())

    save_conversations()

    return jsonify(chat.serialize())

acronyms = load_acronyms(language)
@app.route("/api/acronyms", methods=["GET"])
def get_acronyms():
    return jsonify(acronyms)

if __name__ == "__main__":
    app.run()
