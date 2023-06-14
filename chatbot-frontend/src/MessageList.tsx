import { For, createEffect, createMemo } from "solid-js";
import { ChatMessage } from "./api";
import classes from "./MessageList.module.css";

export type MessageListProps = {
    messages: () => ChatMessage[],
};

const USER_AUTHOR = "user";

/**
 * A list of messages, which automatically scroll to the bottom once a new message is added to the list.
 * Received messages (whose author isn't `USER_AUTHOR`, "user") are put to the left, while sent messages
 * are put to the right.
 **/
export default function MessageList(props: MessageListProps) {
    const lastMessage = createMemo(() => {
        let messages = props.messages();
        return messages[messages.length - 1].id;
    }, undefined, {
        equals(prev, next) {
            return prev === next;
        },
    });
    let container: HTMLElement;

    createEffect(() => {
        // Trigger when lastMessage() changes
        const _ = lastMessage();
        container.scrollTop = container.scrollHeight;
    });

    return (
        <section class={classes.container} ref={(c) => container = c}>
            <ul id="chatbot-message-list" class={classes["message-list"]}>
                <For each={props.messages()}>
                    {(message) => {
                        const received = () => message.author !== USER_AUTHOR;
                        return (
                            <li
                                data-index={message.id}
                                id={`message-${message.id}`}
                                class={[
                                    "chatbot-messages",
                                    received() ? classes["received-message"] : classes["own-message"]
                                ].filter(Boolean).join(" ")}
                            >
                                {message.content}
                            </li>
                        );
                    }}
                </For>
            </ul>
        </section>
    );
}
