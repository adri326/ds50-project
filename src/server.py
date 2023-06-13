from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from chatbot import get_chat_pipeline
from chatcontext import ChatContext, PartialMessage
from ulid import ULID

app = Flask(__name__)
app.config["DEBUG"] = False
CORS(app, resources={r"/api/*": {"origins": "*"}})

# chat_pipeline = get_chat_pipeline()
chat_contexes: dict[str, ChatContext] = dict()

@app.route("/api/chat", methods=["POST"])
@cross_origin()
def create_chat():
    chat = ChatContext()
    chat_contexes[chat.id] = chat

    # TODO: remove, this is for testing purposes only
    message = PartialMessage()
    message.set_content("Hello world", "bot")
    chat.append_message(message.build())

    return jsonify(chat.serialize())

@app.route("/api/chat/<chat_id>/", methods=["POST"])
def post_message(chat_id: str):
    data = request.get_json()
    raise AssertionError("Not implemented yet")

if __name__ == "__main__":
    app.run()
