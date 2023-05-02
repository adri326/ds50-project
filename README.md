# ds50-project

A prototype chatbot for recording health metrics for patients; project for the DS50 ("data science") class

## Running the code

First, clone the repository:

```sh
git clone https://github.com/adri326/ds50-project
cd ds50-project
```

```sh
# Only needs to be run once per user
git lfs install
```

I recommend you use conda to install packages in a different environment.

```sh
# Only the first time
conda env create -f env.yml

# To reload the configuration later on
conda env update -f env.yml
```

> Note: when installing packages, run `conda install <your_package>` and `conda env export > env.yml`

## Project organization

- `src/sentiment.py`: entrypoint for the analysis of message sentiments/intents,
    to try and extract data from messages
- `src/chatbot.py`: entrypoint for the chatbot, which should answer to messages
- `src/server.py`: entrypoint for a server to host an API for the chatbot
- *TODO*
