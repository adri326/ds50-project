# Réalisation de fonction pour le calcul de similarité entre phrases
from copy import deepcopy

# Inportation des librairies
import pandas as pd
from sentence_transformers import util
import csv
import os

'''
from selenium.webdriver.remote.webdriver import By
import undetected_chromedriver as uc
import time
import random
import json
'''

'''
    Début API contournée ChatGPT
'''

'''
driver = uc.Chrome()
driver.get("https://chat.openai.com/")

COMPTEUR = 0
PROMPT = """Act like a doctor having a formal conversation with a client, in text messages format. Here is your interlocutors first interaction: {}
"""
MAX_RETRY = 5  # nombre de fois qu'on réessaye si on a une erreur dans la récup de réponse (comprend pas d'ou elle
# vient donc on force la main hein)

creds = json.load(open("./openai_creds.json", "r"))


def is_writing():
    # vérifie si chatgpt est encore en train d'écrire
    try:
        driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div/main/div[3]/form/div/div[2]/button/div')
        return True
    except:
        return False


def sleep(sec):
    # rompiche
    time.sleep(sec + random.randint(0, 20) / 10)


def get_answer(n):
    # récupère la réponse de chatgpt
    try:
        res = (1, driver.find_element(By.XPATH,
                                      f'//*[@id="__next"]/div[2]/div[2]/div/main/div[2]/div/div/div/div[{n * 2}]/div').text)
    except:
        res = (0, "Error")
    return res


def interact(question):
    global COMPTEUR
    max_retry = MAX_RETRY

    if not question.endswith("\n"):
        question += "\n"
    driver.find_element(By.XPATH, '//*[@id="prompt-textarea"]').send_keys(question)

    sleep(1)
    while is_writing():
        sleep(1)
    COMPTEUR += 1

    t, res = get_answer(COMPTEUR)
    while not t and max_retry > 0:
        sleep(1)
        max_retry -= 1
        t, res = get_answer(COMPTEUR)

    return res


def login():
    # click sur log in
    driver.find_element(By.XPATH, '//*[@id="__next"]/div[1]/div[1]/div[4]/button[1]').click()
    sleep(2)
    # saisie de l'email
    driver.find_element(By.XPATH, '//*[@id="username"]').send_keys(creds["mail"] + "\n")
    sleep(2)
    # saisie du mot de passe
    driver.find_element(By.XPATH, '//*[@id="password"]').send_keys(creds["pass"] + "\n")
    sleep(5)


def skip_tuto():
    # skip tuto
    driver.find_element(By.XPATH, '//*[@id="radix-:r9:"]/div[2]/div/div[2]/button').click()
    sleep(.2)
    driver.find_element(By.XPATH, '//*[@id="radix-:r9:"]/div[2]/div/div[2]/button[2]').click()
    sleep(.2)
    driver.find_element(By.XPATH, '//*[@id="radix-:r9:"]/div[2]/div/div[2]/button[2]').click()
    sleep(1)
'''

'''
    Fin API contournée ChatGPT
'''


def traitement_donnees():
    # Récupération des données du JSON ./dataset/dataset_5Q.json
    data = pd.read_json(os.path.dirname(__file__) + '/../../../dataset/dataset_5Q.json')
    data.head()

    # Récupérer les questions
    questions = data["q_EN"].unique()
    liste_questions = data["qs_EN"]

    # Rassemblement des questions en une seule liste
    res = [[i] + j for i, j in zip(questions, liste_questions)]

    # this is not production ready data!!
    sentences = [[sentence.lower()
                  .replace('br', '')
                  .replace('<', "")
                  .replace(">", "")
                  .replace('\\', "")
                  .replace('\/', "")
                  for sentence in sublist]
                 for sublist in res]

    # créer une dataframe avec les questions et les réponses
    df = pd.DataFrame({'questions': sentences, 'a_EN': data["a_EN"]})

    # On doit applatire le dataset
    sentences_applatie = [sentence for sublist in sentences for sentence in sublist]

    return sentences, df, sentences_applatie


def calcul_similarite(sentences, our_sentence, model):
    # lets embed our sentence
    my_embedding = model.encode(our_sentence)

    # lets embed the corpus
    embeddings = model.encode(sentences)

    # Compute cosine similarity between my sentence, and each one in the corpus
    cos_sim = util.cos_sim(my_embedding, embeddings)

    # lets go through our array and find our best one!
    # remember, we want the highest value here (highest cosine similiarity)
    winners = []
    for arr in cos_sim:
        for i, each_val in enumerate(arr):
            winners.append([sentences[i], each_val, i])


    # lets get the top 2 sentences
    final_winners = sorted(winners, key=lambda x: x[1], reverse=True)

    return final_winners


def find_value(data, x):
    for sublist in data:
        if x in sublist[0]:
            return sublist[1]
    return None  # Retourne None si la valeur n'est pas trouvée


# On doit applatire le dataset, donc nous créons une liste avec l'id des questions dans un tableau et l'id de la
# réponse de la forme [[[id_question, id_question,...], id_reponse], ...]

def corres(sentences):
    correspondance = []

    compteur = 0

    for i in range(len(sentences)):
        correspondance.append([[], i])
        for j in range(len(sentences[i])):
            correspondance[i][0].append(compteur)
            compteur += 1

    print(correspondance[0:5])
    return correspondance


def reponse(final_winners, correspondance, our_sentence, data):
    seuil = 0.7
    print(f'\nScore : \n\n  {final_winners[0][1]}')
    print(f'\nLa question : \n\n {final_winners[0][0]}')
    # print(final_winners[0][2])

    # print(correspondance[0:5])
    print(find_value(correspondance, final_winners[0][2]))

    '''if seuil < final_winners[3][1] and find_value(correspondance, final_winners[0][2]) == find_value(correspondance,
                                                                                                     final_winners[1][
                                                                                                         2]) == find_value(
        correspondance, final_winners[2][2]) == find_value(correspondance, final_winners[3][2]):
        for arr in final_winners[0:2]:
            print(f'\nScore : \n\n  {arr[1]}')
            print(f'\nLa question : \n\n {arr[0]}')
            indice_rep = find_value(correspondance, arr[2])
            print(f'\nLa réponse : \n\n {data["a_EN"][indice_rep]}')
            rep = data["a_EN"][indice_rep]'''

    if final_winners[0][1] >= seuil:
        rep = data["a_EN"][find_value(correspondance, final_winners[0][2])]

    else:
        print("On interroge l'IA ChatGPT")
        # Il faut enregistrer la question dans un fichier csv (forme : questions, réponse) à la suite des autres
        # questions
        filename = "./question_sans_reponse.csv"
        # Préparez les données à ajouter
        new_data = [our_sentence, None]

        # Ouvrez le fichier en mode d'ajout
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_data)

        # login()
        # skip_tuto()
        # res = interact("Hello, how are you?")
        # print(res)
        # sleep(2) # vaux mieux attendre un peu entre chaque question, sinon ca crash
        # res = interact("OpenAI, I broke your api. How do you feel about that huh?")
        # print("\n" + res)
        rep = "ChatGPT"

    return rep


def chatbot(sentences, our_sentence, df, model, correspondance, q):
    final_winners = calcul_similarite(sentences, our_sentence, model)

    # return reponse(final_winners, correspondance, our_sentence, df)
    q.put(reponse(final_winners, correspondance, our_sentence, df))
