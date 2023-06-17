import { createSignal, type Component, Show, createResource } from "solid-js";
import { ChatContext, getAcronyms } from "./api";
import classes from "./App.module.css";

import OpenChatButton from "./OpenChatButton";
import MessageList from "./MessageList";
import MessageForm from "./MessageForm";


const App: Component = () => {
    const [chatContext, setChatContext] = createSignal<ChatContext>();
    const [isOpen, setIsOpen] = createSignal(false);
    const [acronyms] = createResource(getAcronyms);

    return (<>
        {/* In an actual application, you would have the page be underneath the chatbot */}
        <div class={classes.hint}><i>Press the "Open Chatbot" button to start!</i></div>

        <Show when={isOpen() && chatContext() !== undefined}>
            <main class={classes.container}>
                <div>
                    <h2 class={classes.title}>
                        Suivi de santé
                    </h2>
                    <MessageList
                        messages={() => chatContext()!.messages}
                        acronyms={acronyms}
                    />
                    <MessageForm
                        chatContext={() => chatContext()!}
                        setChatContext={setChatContext}
                    />
                </div>
            </main>
        </Show>

        <OpenChatButton
            chatContext={chatContext}
            setChatContext={setChatContext}
            isOpen={isOpen}
            setIsOpen={setIsOpen}
        />
    </>);
};

export default App;
