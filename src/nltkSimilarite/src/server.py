from sentence_transformers import SentenceTransformer

import chatSimi as chat
import get_any_anwser as get
from flask import Flask, jsonify, request
import json
from multiprocessing import Process, Queue

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
    p = Process(target=chat.chatbot, args=(sentences_applatie, message, df, model, correspondance, q))
    p.start()

    result = q.get()

    data_resultat = {"message": result.replace("\n", "<br/>")}
    response = jsonify(data_resultat)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/liste_question', methods=['GET'])
def liste_questions_med():
    tab_questions = get.get_questions()
    reponse = jsonify(tab_questions)
    reponse.headers.add("Access-Control-Allow-Origin", "*")
    return reponse


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    # Importer les données
    sentences, df, sentences_applatie = chat.traitement_donnees()
    correspondance = chat.corres(sentences)

    # Chargement du modèle
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # print(chat.chatbot(sentences, "What is the difference between a data scientist and a data engineer?", df))

    app.run()
