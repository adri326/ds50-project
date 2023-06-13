import { Match, Switch, createSignal } from "solid-js";
import { ChatContext, createChat } from "./api";

export type OpenChatButtonProps = {
    setChatContext: (context: ChatContext) => void,
    chatContext: () => (ChatContext | undefined),
    isOpen: () => boolean,
    setIsOpen: (open: boolean) => void,
}

export default function OpenChatButton(props: OpenChatButtonProps) {
    const [loading, setLoading] = createSignal(false);
    return (
        <button id="chatbot-open-container" title="Open the chatbot" onClick={async () => {
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
        }}>
            <Switch>
                <Match when={!props.isOpen() && !loading()}><i class="fa-solid fa-comments"></i></Match>
                <Match when={props.isOpen()}><i class="fa-solid fa-xmark" id="close-chat-button"></i></Match>
            </Switch>
        </button>
    );
}
