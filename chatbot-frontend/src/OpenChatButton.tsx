import { Match, Switch, createSignal } from "solid-js";
import { ChatContext, createChat } from "./api";
import classes from "./OpenChatButton.module.css";

export type OpenChatButtonProps = {
    setChatContext: (context: ChatContext) => void,
    chatContext: () => (ChatContext | undefined),
    isOpen: () => boolean,
    setIsOpen: (open: boolean) => void,
}

/**
 * A button to open and close the chat box, positioned absolutely on the page.
 **/
export default function OpenChatButton(props: OpenChatButtonProps) {
    const [loading, setLoading] = createSignal(false);
    return (
        <button
            class={[classes.button, props.isOpen() ? classes.open : ""].filter(Boolean).join(" ")}
            title="Open the chatbot"
            onClick={async () => {
                if (props.isOpen()) {
                    props.setIsOpen(false);
                } else {
                    if (props.chatContext() === undefined && !loading()) {
                        try {
                            setLoading(true);
                            const chat = await createChat();
                            props.setChatContext(chat);
                        } finally {
                            setLoading(false);
                        }
                    }
                    props.setIsOpen(true);
                }
            }}
        >
            <Switch>
                <Match when={!props.isOpen() && !loading()}><i class="fa-solid fa-comments"></i></Match>
                <Match when={props.isOpen()}><i class="fa-solid fa-xmark" id="close-chat-button"></i></Match>
            </Switch>
        </button>
    );
}
