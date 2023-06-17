import { createEffect, createSignal } from "solid-js";
import { ChatContext, postMessage } from "./api";
import classes from "./MessageForm.module.css";

export type MessageFormProps = {
    chatContext: () => ChatContext,
    setChatContext: (context: ChatContext) => void,
}

const SendState = {
    Idle: 0,
    InTransit: 1,
    Error: 2,
} as const;
type SendState = typeof SendState[keyof typeof SendState];

// TODO: implement a state machine for locking the sending of messages until a reply is received (or not).
/**
 * An input box to type a message in and a send button.
 * When the button is pressed, the input box is focused again,
 * a message is sent and sending messages becomes impossible until a reply is received.
 **/
export default function MessageForm(props: MessageFormProps) {
    const [content, setContent] = createSignal("");
    const [sendState, setSendState] = createSignal<SendState>(SendState.Idle);
    let input: HTMLInputElement;

    async function sendMessage() {
        if (sendState() !== SendState.InTransit) {
            setSendState(SendState.InTransit);

            input.focus();
            const newChat = props.chatContext().clone();
            const message = content();
            // TODO: send the message
            newChat.messages.push({
                id: Math.random().toString(),
                author: "user",
                content: message,
                time: new Date(),
                metadata: {}
            });
            props.setChatContext(newChat);
            setContent("");

            try {
                const response = await postMessage(newChat, message);
                props.setChatContext(response);
                setSendState(SendState.Idle);
            } catch (err) {
                console.error(err);
                const errorChat = props.chatContext().clone();
                errorChat.messages.push({
                    id: Math.random().toString(),
                    author: "error",
                    content: `Une erreur est survenue lors de l'envoi de votre message, veuillez réessayer`,
                    time: new Date(),
                    metadata: {
                        error: true
                    }
                });
                props.setChatContext(errorChat);
                setSendState(SendState.Error);
            }
        }
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
            }} disabled={sendState() === SendState.InTransit}>
                <i class="fa-solid fa-paper-plane send-icon"></i>
            </button>
        </form>
    );
}
