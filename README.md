# WellnessAgent Chatbot

WellnessAgent Chatbot app utilizes AI/ML to provide psychedelic medicine diagnoses and treatments using llama and langchan to index data. This data is then used with openAI api to provide Generate Pre-Trained Transformed responses.

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

https://github.com/decoderzhub/WellnessAgent/assets/6371329/2faa3420-df5a-472c-a007-bd1a48b2c602
