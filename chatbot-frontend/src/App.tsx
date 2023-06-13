import { createSignal, type Component, createEffect, Show } from 'solid-js';
import { ChatContext, createChat } from './api';
import OpenChatButton from './OpenChatButton';
import MessageList from './MessageList';
import MessageForm from './MessageForm';

const App: Component = () => {
    const [chatContext, setChatContext] = createSignal<ChatContext>();
    const [isOpen, setIsOpen] = createSignal(false);

    return (<>
        <Show when={isOpen() && chatContext() !== undefined}>
            <main class="chatbot-container">
                <div class="chatbot-interface">
                    <h2 class="chatbot-header">
                        Suivi de santé
                    </h2>
                    <section class="chatbot-chat">
                        <MessageList messages={() => chatContext()!.messages} />
                    </section>
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
