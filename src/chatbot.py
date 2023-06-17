from typing import Callable, Iterable, Tuple
from chatcontext import ChatContext, ChatPipeline, FallbackResponse, PartialMessage
from sentiment import load_word2vec_model
from similarity import SimilarityStep, load_questions_from_json
from sentence_transformers import SentenceTransformer
from acronym import load_acronyms

def sentiment_step(evaluate_model: Callable[[str], Iterable[Tuple[str, float]]]) -> Callable[[ChatContext, PartialMessage], None]:
    def evaluate(ctx: ChatContext, msg: PartialMessage):
        values = list(map(lambda pair: pair[1], evaluate_model(ctx.messages[-1].content)))
        msg.set_metadata("sentiment", sum(values) / len(values))

    return evaluate

def get_chat_pipeline(lang: str) -> ChatPipeline:
    pipeline = ChatPipeline()

    pipeline.add_step("similarity", SimilarityStep(
        load_questions_from_json(
            "./dataset/dataset_5Q.json",
            "q_EN" if lang == "en" else "q_FR",
            "qs_EN" if lang == "en" else None,
            "a_EN" if lang == "en" else "a_FR",
        ),
        SentenceTransformer("distiluse-base-multilingual-cased-v1"),
        0.66,
        load_acronyms(lang)
    ))

    pipeline.add_step("sentiment", sentiment_step(load_word2vec_model(6)[0]))

    pipeline.add_step("fallback", FallbackResponse(["Sorry, I couldn't understand you"]))

    return pipeline

if __name__ == "__main__":
    context = ChatContext()
    pipeline = get_chat_pipeline()

    while True:
        message = PartialMessage()
        try:
            message.set_content(input("> "), "user")
        except EOFError:
            break
        answer = pipeline.get_answer(context, message)
        print(answer.serialize())
