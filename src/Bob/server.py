from flask import Flask, jsonify, request
import json
from multiprocessing import Process, Queue

import test as chatbot

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['POST'])
def interogation():
    # Récupérer le corps de la requête
    data = request.data
    # Convertir les données en string
    data_str = data.decode('utf-8')

    # Traiter les données JSON
    data_dict = json.loads(data_str)
    # Récupérer la valeur de la clé 'message'
    message = data_dict['data']
    print(message)

    # Créer une file d'attente pour la valeur de retour
    q = Queue()

    # Exécuter la fonction en parallèle
    p = Process(target=chatbot.pred_class, args=("salut", words, classes, model, q))
    p.start()

    # intents = chatbot.pred_class("salut", words, classes, model)
    result = chatbot.get_response(q.get(), data_all)

    data_resultat = {"message": result}
    response = jsonify(data_resultat)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    model, words, classes = chatbot.setModel()
    data_all = json.loads(open("data.json").read())
    app.run()
