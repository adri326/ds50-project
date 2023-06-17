import { For, Show, createEffect, createMemo, createSignal } from "solid-js";
import { ChatMessage } from "./api";
import classes from "./MessageList.module.css";
import md from "markdown-it";
import { template } from "solid-js/web";

export type MessageListProps = {
    messages: () => ChatMessage[],
    acronyms: () => (Record<string, string> | undefined),
    isLoading: () => boolean,
};

const USER_AUTHOR = "user";

type AcronymHover = {
    x: number,
    y: number,
    value: string,
}

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

    const [acronymHover, setAcronymHover] = createSignal<AcronymHover | undefined>(undefined);

    createEffect(() => {
        // Trigger when lastMessage() changes
        const _ = lastMessage();
        container.scrollTop = container.scrollHeight;
    });

    const renderer = md({
        html: false,
        breaks: true,
    });

    return (
        <section class={classes.container} ref={(c) => container = c}>
            <ul id="chatbot-message-list" class={classes["message-list"]}>
                <For each={props.messages()}>
                    {(message) => {
                        const sentiment = () => {
                            const sentiment = message.metadata["sentiment"];
                            if (typeof sentiment !== "number") return;

                            const sentimentText = (100 * sentiment).toFixed(1).padStart(2, '0') + "%";

                            if (sentiment > 0.5) {
                                return [`color-mix(in srgb, #808080, green ${200 * (sentiment - 0.5)}%`, "Sentiment: " + sentimentText, sentimentText];
                            } else {
                                return [`color-mix(in srgb, #808080, red ${200 * (0.5 - sentiment)}%`, "Sentiment: " + sentimentText, sentimentText];
                            }
                        };
                        const received = () => message.author !== USER_AUTHOR;
                        const isError = () => message.metadata["error"] === true;
                        const rendered = renderer.render(message.content);

                        let item: HTMLDivElement;

                        const renderedResolved = () => {
                            let result = rendered;
                            const acronyms = props.acronyms();
                            if (!acronyms) return result;

                            for (const acronym in acronyms) {
                                result = result.replace(
                                    new RegExp(acronym, "g"),
                                    (value) => '<span class="' + classes.acronym + '">' + value + '</span>'
                                );
                            }

                            return result;
                        };

                        createEffect(() => {
                            const acronyms = props.acronyms();
                            if (!acronyms) return;

                            for (const acronymElement of item.getElementsByClassName(classes.acronym)) {
                                const acronym = acronyms[acronymElement.innerHTML];
                                acronymElement.addEventListener("mouseover", () => {
                                    setAcronymHover({
                                        x: (acronymElement as HTMLElement).offsetLeft + (acronymElement as HTMLElement).offsetWidth / 2,
                                        y: (acronymElement as HTMLElement).offsetTop,
                                        value: acronym,
                                    });
                                });
                                acronymElement.addEventListener("mouseleave", () => {
                                    setAcronymHover(undefined);
                                });
                            }
                            // Find acronyms and attach the hover action on them
                        });

                        return (
                            <li
                                data-index={message.id}
                                id={`message-${message.id}`}
                                class={[
                                    "chatbot-messages",
                                    received() ? classes["received-message"] : classes["own-message"],
                                    isError() ? classes["error-message"] : ""
                                ].filter(Boolean).join(" ")}

                            >
                                <div
                                    class={classes.content}
                                    innerHTML={renderedResolved()}
                                    ref={(l) => item = l}
                                ></div>
                                <div class={classes.metadata}>
                                    <span class={classes.text}>{message.author}</span>
                                    <Show when={sentiment()}>
                                        <div class={classes.sentiment} style={{
                                            "--color": sentiment()![0],
                                        }} title={sentiment()![1]}></div>
                                        <div class={classes.text}>{sentiment()![2]}</div>
                                    </Show>
                                </div>
                            </li>
                        );
                    }}
                </For>
                <Show when={props.isLoading()}><li class={[classes.loading, classes["received-message"]].join(" ")}>
                    <i class="fa-solid fa-ellipsis"></i>
                </li></Show>
            </ul>

            <Show when={acronymHover()}>
                <div class={classes["acronym-hover-message"]} style={{
                    left: `${Math.max(acronymHover()!.x, 100)}px`,
                    top: `${acronymHover()!.y}px`,
                }}>{acronymHover()!.value}</div>
            </Show>
        </section>
    );
}
