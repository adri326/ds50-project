from dotenv import dotenv_values
import openai
import os

from chatcontext import ChatContext, PartialMessage

openai.api_key = dotenv_values(".env.local")["OPENAI_API_KEY"]

class ChatGptStep:
    def __init__(self, lang: str, anxious_threshold: float = 0.25):
        self.initial_message = {
            "role": "system",
            "content": {
                "en": "You are a medical AI assistant named Bob, who focuses on patients with long-term heart conditions. You try to be factual while also being reassuring. Do not try to diagnose the patient, as this task is reserved to the GP the patient has, which he should call if there is anything out of the ordinary.",
                "fr": "Vous êtes une IA assistant médical nommée Bob, qui s'occupe de patients avec des conditions cardiaque de long terme. Vous essayez d'être à la fois factuel et réassurant, tout en restant poli. Il est important que le patient contacte son médecin s'il ressent quelque chose d'hors du commun. Vous n'essayez pas de faire de diagnostic, car cette tâche est réservée au médecin."
            }[lang]
        }

        self.lang = lang
        self.anxious_threshold = anxious_threshold

    def __call__(self, context: ChatContext, message: PartialMessage):
        if message.has_content():
            return

        messages = [self.initial_message]

        for msg in context.messages:
            role = "user" if msg.author == "user" else "assistant"
            content = msg.content
            messages.append({"role": role, "content": content})

        if "sentiment" in message.metadata and message.metadata["sentiment"] < self.anxious_threshold:
            messages.append({
                "role": "system",
                "content": {
                    "en": "Please try to address the patient's worries in a calming yet factual way.",
                    "fr": "Le patient semble être anxieux, veuillez être calmant tout en restant factuel."
                }[self.lang]
            })

        # try:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
        )

        message.set_content(chat_completion.choices[0].message.content, "chatgpt")
        # except:
        #     pass
