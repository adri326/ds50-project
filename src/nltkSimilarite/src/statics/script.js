const chat = document.getElementById("chatbot-chat");
const sendButton = document.getElementById("chatbot-new-message-send-button");
const sendIcon = document.getElementById("send-icon");

$("#chatbot-open-container").click(function () {
    $("#open-chat-button").toggle(200);
    $("#close-chat-button").toggle(200);
    $("#chatbot-container").fadeToggle(200);
});

sendButton.addEventListener("click", function () {
    if (!sendButton.classList.contains("loading")) {
        newInput();
    }
});

document.getElementById("chatbot-input").addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !sendButton.classList.contains("loading")) {
        newInput();
    }
});

function newInput() {
    const newText = document.getElementById("chatbot-input").value;
    console.log(newText);
    if (newText !== "") {
        document.getElementById("chatbot-input").value = "";
        addMessage("sent", newText);
        generateResponse(newText, newText);
    }
}

function addMessage(type, text) {
    let messageDiv = document.createElement("div");
    let responseText = document.createElement("p");
    responseText.innerHTML = text;

    if (type === "sent") {
        messageDiv.classList.add("chatbot-messages", "chatbot-sent-messages");
    } else if (type === "received") {
        messageDiv.classList.add("chatbot-messages", "chatbot-received-messages");
    }

    messageDiv.appendChild(responseText);
    chat.prepend(messageDiv);
}

function generateResponse(prompt, text) {
    sendIcon.classList.remove("fa-solid", "fa-paper-plane");
    sendIcon.classList.add("fa-solid", "fa-spinner", "fa-spin");
    sendButton.classList.add("loading"); // On ajoute la classe "loading" au bouton d'envoi

    console.log("on intéroge l'API : ", text);
    const data = {data: text};
    const config = {
        method: "POST",
        body: JSON.stringify(data)
    }
    fetch('http://127.0.0.1:5000/', config)
        .then(response => response.json()
            .then(data => {
                sendIcon.classList.remove("fa-solid", "fa-spinner", "fa-spin");
                sendIcon.classList.add("fa-solid", "fa-paper-plane");
                sendButton.classList.remove("loading"); // On supprime la classe "loading" du bouton d'envoi
                addMessage("received", data.message);
            })
            .catch(error => {
                sendIcon.classList.remove("fa-solid", "fa-spinner", "fa-spin");
                sendIcon.classList.add("fa-solid", "fa-paper-plane");
                sendButton.classList.remove("loading"); // On supprime la classe "loading" du bouton d'envoi
                addMessage("received", 'Une erreur s\'est produite : ' + error);
            }))
        .catch(error => {
            console.log('Une erreur s\'est produite : ' + error);
        });

    console.log("L'API répond : ", "Great to hear that!");
}
