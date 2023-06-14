import { createEffect, createSignal } from "solid-js";
import { ChatContext } from "./api";
import classes from "./MessageForm.module.css";

export type MessageFormProps = {
    chatContext: () => ChatContext,
    setChatContext: (context: ChatContext) => void,
}

// TODO: implement a state machine for locking the sending of messages until a reply is received (or not).
/**
 * An input box to type a message in and a send button.
 * When the button is pressed, the input box is focused again,
 * a message is sent and sending messages becomes impossible until a reply is received.
 **/
export default function MessageForm(props: MessageFormProps) {
    const [content, setContent] = createSignal("");
    let input: HTMLInputElement;

    async function sendMessage() {
        input.focus();
        const newChat = props.chatContext().clone();
        // TODO: send the message
        newChat.messages.push({
            id: Math.random().toString(),
            author: "user",
            content: content(),
            time: new Date(),
            metadata: {}
        });
        props.setChatContext(newChat);
        setContent("");
    }

    createEffect(() => {
        input.focus();
    });

    return (
        <form class={classes.form} onSubmit={(event) => {
            event.preventDefault();

            if (content().trim()) {
                sendMessage();
            }

            return false;
        }}>
            <input
                ref={(i) => input = i}
                class={classes.input}
                type="text"
                name="chatbot-input"
                placeholder="Type a message here..."
                value={content()}
                onChange={(e) => setContent(e.target.value)}
            />
            <button type="submit" class={classes.send} title="Send message" onClick={() => {
                input.focus();
            }}>
                <i class="fa-solid fa-paper-plane send-icon"></i>
            </button>
        </form>
    );
}
