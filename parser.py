import pandas as pd
import os

def parse():
    tout = {
        "questions": [],
        "answers": [],
        "url": [],
        "name": []
    }

    for i in os.listdir("./data/"):
        with open("./data/" + i, "r", encoding="utf-8") as f:
            fichier = f.read()
        
        fichier = fichier.split("\n\n\n")
        for j in fichier[1:]:
            j = j.split("\n")
            tout["questions"].append(j[0])
            tout["answers"].append("\n".join(j[1:]))
            tout["url"].append(fichier[0].replace("#URL:", ""))
            tout["name"].append(i.replace(".txt", "").replace("_", " "))

    return pd.DataFrame(tout)

