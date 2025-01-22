import os
import json

import google.generativeai as genai

#getting the working directory
working_directory=os.path.dirname(os.path.abspath(__file__))

config_file_path=f"{working_directory}/config.json"
config_data=json.load(open(config_file_path))

# Loading the api key

GOOGLE_API_KEY=config_data["GOOGLE_API_KEY"]

# configuring google.generativeai with api key

genai.configure(api_key=GOOGLE_API_KEY)

# creating a function to load the gemini pro model for chatbot and return it
def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model

# function for image captioning
def gemini_pro_vision_response(prompt, image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([prompt, image])
    result = response.text
    return result

# function to get embedding for text
def embedding_model_response(input_text):
    embedding_model = "models/text-embedding-004"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list=embedding["embedding"]
    return embedding_list

# function to get a response from gemini-pro llm
def gemini_pro_response(user_prompt):
    gemini_pro_model=genai.GenerativeModel("gemini-pro")
    response=gemini_pro_model.generate_content(user_prompt)
    result=response.text
    return result

