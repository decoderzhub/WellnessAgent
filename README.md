# WellnessAgent Chatbot

## Getting Started

Make sure to git clone this repo with:

```git clone https://github.com/decoderzhub/WellnessAgent.git ```

Then cd into the directory:

```cd WellnessAgent```

First you will need to install all of the necessary libraries pip install the requirements.txt

```pip3 install -r requirements.txt```

## Configure OpenAI Key

To set the OpenAI secret key go to the `.streamlit/secrets.toml` file and set the `api_secret`:

```api_secret="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"```

## Start Streamlit Application

```streamlit run main.py```