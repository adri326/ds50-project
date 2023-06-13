import { createSignal } from "solid-js";
import { ChatContext } from "./api";

export type MessageFormProps = {
    chatContext: () => ChatContext,
    setChatContext: (context: ChatContext) => void,
}

export default function MessageForm(props: MessageFormProps) {
    const [content, setContent] = createSignal("");

    async function sendMessage() {
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

    return (
        <form class="chatbot-input" onSubmit={(event) => {
            event.preventDefault();

            if (content().trim()) {
                sendMessage();
            }

            return false;
        }}>
            <div class="chatbot-input-container">
                <input
                    type="text"
                    name="chatbot-input"
                    placeholder="Type a message here..."
                    value={content()}
                    onChange={(e) => setContent(e.target.value)}
                />
            </div>
            <button type="submit" id="chatbot-send-button" title="Send message">
                <i class="fa-solid fa-paper-plane send-icon"></i>
            </button>
        </form>
    );
}
