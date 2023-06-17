from typing import Callable, Iterable, Tuple
import re

import pandas as pd
from sentence_transformers import util, SentenceTransformer
from chatcontext import ChatContext, PartialMessage

class SimilarityStep:
    def __init__(self, pairs: pd.DataFrame, model: SentenceTransformer, threshold: float, acronyms: dict[str, str] | None = None):
        # A dataframe containing the questions (in the `questions` column) and their corresponding answer (in the `answer` column)
        self.pairs = pairs

        # The model used to compute the similarity of sentences
        self.model = model

        # The threshold for determining whether an answer should be given
        self.threshold = threshold

        # A dictionary mapping the questions to the index of their answer
        self.question_indices = dict()
        for index, row in self.pairs.iterrows():
            if type(row["questions"]) == str:
                self.question_indices[row["questions"]] = index

            for question in row["questions"]:
                self.question_indices[question] = index


        # A flattened dataframe containing all questions
        self.questions_flattened = self.pairs["questions"].explode().to_frame("question").reset_index()

        # Augment the questions with acronyms
        if acronyms != None:
            additional_questions = []

            def apply_acronym(question: str, src: str, target: str, case: bool):
                if case:
                    replaced = question.replace(src, target)
                else:
                    replaced = re.sub(re.compile(re.escape(src), re.IGNORECASE), target, question)

                if replaced != question:
                    additional_questions.append(replaced)
                    self.question_indices[replaced] = self.question_indices[question]

            for i, question in self.questions_flattened.iterrows():
                for acronym, definition in acronyms.items():
                    apply_acronym(question["question"], acronym, definition, True)
                    apply_acronym(question["question"], definition, acronym, False)

            additional_questions = pd.DataFrame({"question": additional_questions})
            self.questions_flattened = pd.concat([self.questions_flattened, additional_questions], ignore_index=True)
            # Please just reset the index already ;w;
            self.questions_flattened = self.questions_flattened["question"].to_frame("question").reset_index(drop=True)

        # A tensor containing the embedding of all questions; may take a bit of time to compute
        self.question_embeddings = model.encode(self.questions_flattened["question"].to_list())

    def get_answer(self, answer_id: int) -> str | None:
        return self.pairs["answer"][answer_id]

    # Finds the the most similar question matching `prompt`.
    # Returns a tuple containing the similarity score, the matched question and the corresponding answer
    def get_most_similar(self, prompt: str) -> Tuple[float, str, int] | None:
        prompt_embedding = self.model.encode(prompt)
        similarity = util.cos_sim(prompt_embedding, self.question_embeddings)[0]

        best = max(enumerate(similarity), default=None, key=lambda pair: pair[1])

        if best == None:
            return None

        best_index, best_score = best
        best_question = self.questions_flattened["question"][best_index]

        return (
            best_score.item(),
            best_question,
            self.question_indices[best_question],
        )

    # Interface required to be used in a `ChatPipeline`
    def __call__(self, context: ChatContext, message: PartialMessage):
        if message.has_content():
            return

        most_similar = self.get_most_similar(context.messages[-1].content)
        if most_similar == None:
            return

        # Don't set the answer if the score is below the threshold
        if most_similar[0] < self.threshold:
            return

        answer = self.get_answer(most_similar[2])

        message.set_content(answer, "similarity")

def load_questions_from_json(
    path: str,
    base_question_column: str = "q_EN",
    alt_questions_column: str | None = "qs_EN",
    answer_column: str = "a_EN",
) -> pd.DataFrame:
    raw = pd.read_json(path)

    pairs = pd.DataFrame()
    pairs["questions"] = raw[base_question_column].map(lambda question: [question]).astype("object")
    pairs["answer"] = raw[answer_column]

    if alt_questions_column != None:
        pairs["questions"] = pairs["questions"] + raw[alt_questions_column]

    return pairs

if __name__ == "__main__":
    from chatcontext import ChatPipeline

    model = SentenceTransformer("all-MiniLM-L6-v2")
    step = SimilarityStep(
        load_questions_from_json("./dataset/dataset_5Q.json"),
        model,
        0.75
    )

    pipeline = ChatPipeline()
    pipeline.add_step("similarity", step)

    context = ChatContext()
    message = PartialMessage()
    message.set_content("Is my heart rate too fast?", "user")

    print(pipeline.get_answer(context, message.build()).content)
