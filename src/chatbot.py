# TODO :)

from typing import Callable, Iterable, Tuple
from chatcontext import ChatContext, ChatPipeline, FallbackResponse, PartialMessage
from sentiment import load_word2vec_model

def sentiment_step(evaluate_model: Callable[[str], Iterable[Tuple[str, float]]]) -> Callable[[ChatContext, PartialMessage], None]:
    def evaluate(ctx: ChatContext, msg: PartialMessage):
        values = list(map(lambda pair: pair[1], evaluate_model(ctx.messages[-1].content)))
        msg.set_metadata("sentiment", sum(values) / len(values))

    return evaluate

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from similarity import SimilarityStep, load_questions_from_json

    pipeline = ChatPipeline()
    pipeline.add_step("sentiment", sentiment_step(load_word2vec_model(6)[0]))

    pipeline.add_step("similarity", SimilarityStep(
        load_questions_from_json("./dataset/dataset_5Q.json"),
        SentenceTransformer("all-MiniLM-L6-v2"),
        0.75
    ))

    pipeline.add_step("fallback", FallbackResponse(["Sorry, I couldn't understand you"]))

    context = ChatContext()

    while True:
        message = PartialMessage()
        try:
            message.set_content(input("> "), "user")
        except EOFError:
            break
        answer = pipeline.get_answer(context, message)
        print(answer.serialize())
