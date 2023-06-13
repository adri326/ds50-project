import { For } from "solid-js";
import { ChatMessage } from "./api";

export type MessageListProps = {
    messages: () => ChatMessage[],
};

const USER_AUTHOR = "user";

export default function MessageList(props: MessageListProps) {
    return (
        <ul id="chatbot-message-list">
            <For each={props.messages()}>
                {(message) => {
                    const received = message.author !== USER_AUTHOR;
                    return (
                        <li
                            data-index={message.id}
                            class={[
                                "chatbot-messages",
                                received ? "chatbot-received-messages" : "chatbot-own-messages"
                            ].filter(Boolean).join(" ")}
                        >
                            {message.content}
                        </li>
                    );
                }}
            </For>
        </ul>
    );
}
