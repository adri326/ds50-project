from datetime import date
from typing import Callable, Union, Tuple, List, Any
from enum import Enum
import random
import unittest

from ulid import ULID

class Message:
    id: str
    author: str
    content: str
    time: date
    metadata: dict

    def __init__(self, id: str, author: str, content: str, time: date, metadata: dict) -> None:
        self.id = id
        self.author = author
        self.content = content
        self.time = time
        self.metadata = metadata

    def serialize(self) -> dict:
        result = dict()
        result["id"] = self.id
        result["author"] = self.author
        result["content"] = self.content
        result["time"] = self.time.isoformat()
        result["metadata"] = self.metadata

        return result

    @staticmethod
    def deserialize(raw: dict) -> "Message":
        id = str(raw["id"])
        author = str(raw["author"])
        content = str(raw["content"])
        time = date.fromisoformat(raw["time"])
        metadata = dict(raw["metadata"])

        return Message(id, author, content, time, metadata)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Message):
            return False
        return other.id == self.id \
            and other.author == self.author \
            and other.time == self.time \
            and other.metadata == self.metadata \
            and other.content == self.content


class PartialMessage:
    id: str
    author: Union[str, None]
    content: Union[str, None]
    time: date
    metadata: dict

    def __init__(self) -> None:
        self.id = str(ULID())
        self.time = date.today()
        self.author = None
        self.content = None
        self.metadata = dict()

    def set_content(self, content: str, author: str):
        self.content = content
        self.author = author

    def has_content(self) -> bool:
        return not (self.author == None and self.content == None)

    def set_metadata(self, key: str, data: Any):
        self.metadata[key] = data

    def build(self) -> Union[Message, None]:
        if self.author == None or self.content == None:
            return None

        return Message(self.id, self.author, self.content, self.time, self.metadata)

class ChatState(Enum):
    Idle = 0
    Answering = 1

class ChatContext:
    def __init__(self, id = str(ULID())) -> None:
        self.id: str = id
        self.messages: List[Message] = []
        self._response = None
        self.state: ChatState = ChatState.Idle

    def append_message(self, message: Message):
        self.messages.append(message)

    def serialize(self) -> dict:
        result = dict()
        result["id"] = self.id
        result["messages"] = list(map(lambda msg: msg.serialize(), self.messages))

        return result

    def deserialize(raw: dict) -> "ChatContext":
        result = ChatContext(str(raw["id"]))
        result.messages = list(map(lambda raw_msg: Message.deserialize(raw_msg), raw["messages"]))
        result.state = ChatState.Idle

        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChatContext):
            return False

        return self.id == other.id and self.messages == other.messages

class ChatPipeline:
    steps: List[Tuple[str, Callable[[ChatContext, PartialMessage], None]]]

    def __init__(self) -> None:
        self.steps = []

    def add_step(self, name: str, callback: Callable[[ChatContext, PartialMessage], None]):
        self.steps.append((name, callback))

    # TODO: make async
    def get_answer(self, context: ChatContext, user_message: Union[Message, None] = None) -> Message:
        new_message = PartialMessage()
        if context.state == ChatState.Answering:
            raise AssertionError(f"ChatContext is already answering!")

        context.state = ChatState.Answering

        try:
            if user_message != None:
                context.append_message(user_message)

            for name, step in self.steps:
                try:
                    step(context, new_message)
                except Exception as err:
                    err.args = (f"Error while processing step '{name}'", *err.args)
                    raise

            message = new_message.build()
            if message == None:
                raise AssertionError(f"No step in ChatPipeline populated the message!")

            context.append_message(message)
            return message
        finally:
            context.state = ChatState.Idle

class FallbackResponse:
    responses: List[str]

    def __init__(self, responses: List[str] = []) -> None:
        self.responses = responses

    def __call__(self, ctx: ChatContext, user_message: PartialMessage) -> None:
        if not user_message.has_content():
            user_message.set_content(self.responses[random.randint(0, len(self.responses) - 1)], "fallback")

class TestChat(unittest.TestCase):
    def test_response(self):
        pipeline = ChatPipeline()
        pipeline.add_step("fallback", FallbackResponse(["I did not catch that, sorry"]))

        context = ChatContext()
        message = pipeline.get_answer(context)
        self.assertEqual(message.content, "I did not catch that, sorry")
        self.assertEqual(context.messages[0], message)

    def test_mutex(self):
        pipeline = ChatPipeline()
        def test(ctx: ChatContext, msg: PartialMessage):
            self.assertEqual(ctx.state, ChatState.Answering)
        pipeline.add_step("test", test)
        pipeline.add_step("fallback", FallbackResponse(["Fallback"]))

        context = ChatContext()
        self.assertEqual(context.state, ChatState.Idle)

        pipeline.get_answer(context)
        self.assertEqual(context.state, ChatState.Idle)

    def test_mutex_err(self):
        invalid_pipeline = ChatPipeline()
        def try_get_answer_recursively(ctx: ChatContext, msg: PartialMessage):
            invalid_pipeline.get_answer(ctx)
        invalid_pipeline.add_step("test", try_get_answer_recursively)
        invalid_pipeline.add_step("fallback", FallbackResponse(["Fallback"]))

        context = ChatContext()
        self.assertRaises(AssertionError, lambda: invalid_pipeline.get_answer(context))

    def test_serde(self):
        import json

        context = ChatContext()
        msg = PartialMessage()
        msg.set_content("Hello world", "me")
        msg.set_metadata("sentiment", 0.5)
        context.append_message(msg.build())

        serialized = context.serialize()
        deserialized = ChatContext.deserialize(serialized)

        self.assertEqual(deserialized, context)

        json_deserialized = ChatContext.deserialize(json.loads(json.dumps(serialized)))
        self.assertEqual(json_deserialized, context)

if __name__ == "__main__":
    unittest.main()
