import pandas
import os

def load_acronyms(lang: str) -> dict[str, str]:
    raw = pandas.read_json(os.path.dirname(__file__) + "/../dataset/acro.json")

    acronym_column = "EN" if lang == "en" else "FR"
    expanded_column = "EN_expand" if lang == "en" else "FR_expand"

    result = dict()
    for index, row in raw.iterrows():
        # Only keep acronyms for which we have a definition
        if len(row[expanded_column]) > 0:
            result[row[acronym_column]] = row[expanded_column]

    return result
