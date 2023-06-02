# Pour récupérer les questions sans réponse du fichier ./questions_sans_reponse.csv

import pandas as pd


def get_questions():
    questions = []
    reader = pd.read_csv('./question_sans_reponse.csv', header=None, names=['questions', 'reponses'])
    for index, row in reader.iterrows():
        questions.append(row['questions'])
    return questions


print(get_questions())
