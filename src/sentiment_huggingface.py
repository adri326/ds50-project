from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

from chatcontext import ChatContext, PartialMessage

class HuggingFaceSentiment:
    def __init__(self):
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def __call__(self, context: ChatContext, message: PartialMessage):
        encoded_input = self.tokenizer(context.messages[-1].content, return_tensors="pt")
        result = softmax(self.model(**encoded_input)[0][0].detach().numpy())
        scores = dict()
        for index, score in enumerate(result):
            scores[self.config.id2label[index]] = score

        print(scores)
        final_score = (scores["positive"] - scores["negative"]) * 0.5 + 0.5

        message.set_metadata("sentiment", final_score)
