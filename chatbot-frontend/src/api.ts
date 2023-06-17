const API_BASE_URL = (import.meta.env["VITE_API_URL"] as string) ?? "http://localhost:5000/api/";

export type ChatMessage = {
    id: string,
    author: string,
    content: string,
    time: Date,
    metadata: Record<string, unknown>,
}

export class ChatContext {
    constructor(
        readonly id: string,
        readonly messages: ChatMessage[],
    ) {}

    clone(): ChatContext {
        return new ChatContext(this.id, this.messages.slice());
    }
};

export async function createChat(): Promise<ChatContext> {
    return (
        fetch(API_BASE_URL + "chat", {
            method: "POST"
        })
        .then((res) => res.json())
        .then(serializeContext)
    );
}

export async function postMessage(context: ChatContext, message: string): Promise<ChatContext> {
    return (
        fetch(API_BASE_URL + "chat/" + context.id, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message
            })
        })
        .then((res) => res.json())
        .then(serializeContext)
    );
}

export async function getAcronyms(): Promise<Record<string, string>> {
    return (
        fetch(API_BASE_URL + "acronyms")
        .then((res) => res.json())
        .then((res: unknown) => {
            if (typeof res !== "object" || !res) {
                throw new Error("Assertion error: expected response to be an object");
            }

            for (const key in res) {
                const value = (res as Record<string, unknown>)[key];
                if (typeof value !== "string") {
                    throw new Error("Assertion error: expected response to be an object of strings, got " + value);
                }
            }

            return res as Record<string, string>;
        })
    );
}

// One possible optimization for this would be to have a global cache of known message IDs,
// and pull from that cache
export function serializeContext(rawContext: unknown): ChatContext {
    if (typeof rawContext !== "object" || rawContext === null) {
        throw new Error("Assertion error: expected response to be an object");
    }

    if (!("id" in rawContext) || typeof rawContext["id"] !== "string") {
        throw new Error("Assertion error: expected 'id' to be string");
    }

    if (!("messages" in rawContext) || !Array.isArray(rawContext["messages"])) {
        throw new Error("Assertion error: expected 'messages' to be an array");
    }

    const messages: ChatMessage[] = [];
    const requiredKeys = ["id", "author", "content", "time"] as const;

    for (const rawMessage of rawContext["messages"]) {
        for (const key of requiredKeys) {
            if (typeof rawMessage[key] !== "string") {
                throw new Error(`Assertion error: expected 'messages[x].${key}' to be a string`);
            }
        }
        if (typeof rawMessage["metadata"] !== "object" || rawMessage["metadata"] === null || Array.isArray(rawMessage["metadata"])) {
            throw new Error("Assertion error: expected 'messages[x].metadata' to be an object");
        }

        messages.push({
            id: rawMessage["id"],
            author: rawMessage["author"],
            content: rawMessage["content"],
            time: new Date(rawMessage["time"]),
            metadata: rawMessage["metadata"],
        });
    }

    return new ChatContext(rawContext["id"], messages);
}
