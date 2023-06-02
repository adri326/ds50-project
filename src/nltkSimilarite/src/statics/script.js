const chat = document.getElementById("chatbot-chat");


$("#chatbot-open-container").click(function () {
    $("#open-chat-button").toggle(200);
    $("#close-chat-button").toggle(200);
    $("#chatbot-container").fadeToggle(200);
});

document.getElementById("chatbot-new-message-send-button").addEventListener("click", newInput);

document.getElementById("chatbot-input").addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        newInput();
    }
});

function newInput() {
    newText = document.getElementById("chatbot-input").value;
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
    responseText.appendChild(document.createTextNode(text));

    if (type === "sent") {
        messageDiv.classList.add("chatbot-messages", "chatbot-sent-messages");
    } else if (type === "received") {
        messageDiv.classList.add("chatbot-messages", "chatbot-received-messages");
    }

    messageDiv.appendChild(responseText);
    chat.prepend(messageDiv);
}


function generateResponse(prompt, text) {
    // Here you can add your answer-generating code
    console.log("on intéroge l'API : ", text);
    //L'API est appelée ici avec AJAX
    const data = {data: text};
    const config = {
        method: "POST",
        body: JSON.stringify(data)
    }
    fetch('http://127.0.0.1:5000/', config)
        .then(response => response.json()
            .then(data => (addMessage("received", data.message)))
            .catch(error => {addMessage("received", 'Une erreur s\'est produite : ' + error)}))
        .catch(error => {
            console.log('Une erreur s\'est produite : ' + error);
        });

    //L'API répond ici
    console.log("L'API répond : ", "Great to hear that!");
}