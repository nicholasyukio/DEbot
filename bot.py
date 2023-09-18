# DEbot, a OpenAI API based Telegram bot which interprets a student's doubts 
# and then searchs the course (Dominio Eletrico) lesson list to recommend the
# best lessons according to doubt extracted out of the messages sent by the student
# Author: Nicholas Yukio Menezes Sugimoto
# v.1.0 17 September 2023

# Imports section
import os
import openai
import telebot
import requests
import json
from scipy.spatial import distance
from dotenv import load_dotenv
from unidecode import unidecode
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

# Standards
max_messages_context = 10

# Keys section
load_dotenv()
bot = telebot.TeleBot(os.getenv('TELEGRAM_API_TOKEN'))
openai.api_key = os.getenv('OPENAI_KEY')

# Modules names
modules_names = ["1: Conceitos Básicos", 
                 "2: Componentes Eletrônicos", 
                 "3: Análise Básica de Circuitos", 
                 "4: Circuitos de Primeira Ordem",
                 "5: Circuitos de Segunda Ordem",
                 "6: Circuitos em Corrente Alternada",
                 "7: Circuitos Trifásicos",
                 "8: Análise Avançada de Circuitos",
                 "9: Semicondutores",
                 "10: Circuitos Analógicos",
                 "11: Circuitos Digitais",
                 "Domínio Elétrico Labs" ]

# Opening JSON file and retrieving data
def get_data_from_json_file(filename):
    f = open(filename, 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    return data

# Matching keywords
def matching_keywords_with_modules(doubt: str):
    unwanted_characters = "!@#$%^&*()_+:;'<>?,./|1234567890"
    for character in unwanted_characters:
        doubt = doubt.replace(character, '')
    doubt = doubt.lower()
    doubt = doubt.split()
    keywords_data = get_data_from_json_file("keywords.json")
    i = 0
    present_keywords = []
    #Receiving keywords from modules 1 to 11
    for i in range(11):
        if i < 9:
            keywords = keywords_data["modulo_0"+str(i+1)]
        else:  
            keywords = keywords_data["modulo_"+str(i+1)]
        keywords_tmp = []
        for keyword in keywords:
            keywords_tmp.append(keyword.lower())
        keywords = keywords_tmp
        present_keywords.append([s for s in doubt if s in keywords])
    #Receiving keywords from module DE Labs
    keywords = keywords_data["DE_labs"]
    keywords = [keyword.lower() for keyword in keywords]
    present_keywords.append([s for s in doubt if s in keywords])
    #print("Present keywords per module: ", present_keywords)
    return present_keywords

# Selecting a list of modules for lesson search
def select_modules_for_search(present_keywords: list):
    m = len(present_keywords)
    i = 0
    modules = []
    max_len = 0
    for i in range(m):
        if len(present_keywords[i]) > max_len:
            modules = []
            modules.append(i)
            max_len = len(present_keywords[i])
        elif len(present_keywords[i]) == max_len and max_len != 0:
            modules.append(i)
    #print("Value of i_max: ", modules)
    return modules, max_len

# Outputs the name of the module by its index
def module_index_to_filename(index: int):
    if index < 9:
        filename = "modulo_0"+str(index+1)+".json"
    elif index < 11:
        filename = "modulo_"+str(index+1)+".json"
    else:
        filename = "DE_Labs.json"
    return filename

# Embedding recommendation
def recommendations_from_strings(filename, doubt, model="text-embedding-ada-002"):
    """Return nearest neighbors of a given string."""
    # get embeddings for all strings
    data_file=get_data_from_json_file(filename)
    strings = []
    for i in range(len(data_file["aulas"])):
        strings.append(data_file["aulas"][i]["lesson"])
    # get the embedding of the source string
    query_embedding = get_embedding(doubt)
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    embeddings = [get_embedding(string) for string in strings]
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric='cosine')
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    #print(indices_of_nearest_neighbors)
    recommended_lesson = data_file["aulas"][indices_of_nearest_neighbors[0]]["lesson"]
    duration = data_file["aulas"][indices_of_nearest_neighbors[0]]["duration"]
    return recommended_lesson, duration

# Generates a link for a lesson based on its title
def recommeded_lesson_to_link(index: int, recommended_lesson: str):
    unwanted_characters = "!@#$%^&*()_+:;'<>?,./|"
    for character in unwanted_characters:
        recommended_lesson = recommended_lesson.replace(character, '')
    recommended_lesson = recommended_lesson.replace(' ', '-')
    recommended_lesson = recommended_lesson.lower()
    recommended_lesson = recommended_lesson.replace('á', 'a')
    recommended_lesson = recommended_lesson.replace('à', 'a')
    recommended_lesson = recommended_lesson.replace('â', 'a')
    recommended_lesson = recommended_lesson.replace('ã', 'a')
    recommended_lesson = recommended_lesson.replace('é', 'e')
    recommended_lesson = recommended_lesson.replace('ê', 'e')
    recommended_lesson = recommended_lesson.replace('í', 'i')
    recommended_lesson = recommended_lesson.replace('ó', 'o')
    recommended_lesson = recommended_lesson.replace('ô', 'o')
    recommended_lesson = recommended_lesson.replace('õ', 'o')
    recommended_lesson = recommended_lesson.replace('ú', 'u')
    recommended_lesson = recommended_lesson.replace('ç', 'c')
    #print(recommended_lesson)
    if index < 11:
        link = "https://dominioeletrico.com.br/courses/dominio-eletrico/lessons/"+recommended_lesson+"/"
    elif index == 11:
        link = "https://dominioeletrico.com.br/courses/dominio-eletrico-labs/lessons/"+recommended_lesson+"/"
    return link

# OpenAI operation section

# This is the context, which is continuously given to the model so it can produce a response
context = [ {'role':'system', 'content':"""
You are DE Bot, or Dominio Eletrico Bot, and your function is\
to help students to learn the subject of electric circuits.\
After an user starts a chat, you greet the student and say that you work
to help Prof. Nicholas Yukio.\
Then you ask what are the doubts the student has on the subject to help\
with their learning.\
If the student keeps sending gibberish text instead of writing a doubt \
on electric circuits, do not repeat the same initial greeting message. \
Instead, ask using another phrase if there is anyway you could be helpful.\
You can even ask how the student is, using a playful tone.\
However, whenever the student presents a doubt, your mission is not to \
explain the concepts to them, but rather to recommend a lesson from the \
Domínio Elétrico course. However, the lesson recommendation task itself \
should not be done by you. Instead, this task will be done by a separate section of code.\
Be careful to ensure that your responses are no more than 200 caracters long.\
Unless clearly stated otherwise by the user, you should answer in Brazilian Portuguese.
"""} ]  # accumulate messages

# Gets a response from the model for a single prompt
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# Gets a response from the model for a whole context
def get_completion_from_messages(messages, model="gpt-3.5-turbo-16k", temperature=0.2):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# Generates a text to recommend a lesson, without calling the OpenAI model (used preferred in this code)
def recommend_lesson_dumb(recommended_lesson, duration, module, link):
    response_recommended_lesson = f"""
    Recomendo a seguinte aula:\n
    {recommended_lesson},\n que está no módulo {module}, e tem duração {duration}.\n
    Link para acessar a aula: {link}\n
    Talvez seja bom dar uma olhada também nas outras aulas do módulo {module}. Bons estudos!
    """
    return response_recommended_lesson

# Generates a text to recommend a lesson, calling the OpenAI model
def recommend_lesson(recommended_lesson, duration, module, link):
    prompt = f"""
    Recommend to the user the lesson whose name, module and duration are found below,\
    providing in the same response the link to the lesson found below.\
    Remember to use Brazilian Portuguese, unless the user clearly\
    stated their wish to use another language.\
    Make it briefly, with no more than 140 characters.

    Lesson to be recommended: '''{recommended_lesson}'''
    Duration of the lesson: '''{duration}'''
    Module of the lesson: '''{module}'''
    Link to the lesson: '''{link}'''
    """
    response_recommended_lesson = get_completion(prompt)
    return response_recommended_lesson

# Reads a message sent by the user and extracts a possible doubt on electric circuits
def extract_doubt(text):
    prompt = f"""
    Read the text below and search for possible doubts the user has about\
    electric circuits. Remember that the user will most probably type the\
    doubt in Brazilian Portuguese, so look for portuguese words which indicate\
    doubt, difficulty, saying that the user does not understand or asking how
    to calculate something. Below you have some examples of messages, and the\
    doubt present as you should output:\
    Message: "Eu não entendo análise nodal", Doubt: "análise nodal"\
    Message: "Tenho dificuldade em entender os conceitos de fasor e impedância", Doubt: "fasor e impedância"\
    Message: "Não consigo fazer análise de circuitos com transformada de Laplace", Doubt: "análise de circuitos com transformada de laplace"\
    Message: "Como eu calculo a expressão da tensão no capacitor em um circuito RC?", Doubt: "expressão da tensão no capacitor em circuito rc"\
    Message: "Sempre fico com dúvida quando em coeficientes da série de Fourier", Doubt: "série de fourier"\
    
    Make sure to correct possible typos before generating response.
    
    If any doubt is found, summarize and reply in no more\
    than 8 words, with all letters in lower case, without any dots or commas. If no doubt is found, just answer with "None"\
    If you detect the user is expressing some doubt but you it is not clear to you, just answer with "Unsure"\

    Review text: '''{text}'''
    """
    try:
        response_extract_doubt = get_completion(prompt)
    except:
        response_extract_doubt = "balance_depleted"
        #response_extract_doubt = "Estou em greve. Retornarei ao trabalho assim que as condições na OpenAI forem normalizadas."
    return response_extract_doubt

# Telegram bot operation section

# This is the main function which comes into execution when a new message is received by the bot
@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    response_as_recommendation = False
    responses_List = []
    while len(context) > max_messages_context:
        context.pop(1)
    context.append({'role':'user', 'content':f"{message}"})
    doubt = extract_doubt(message)
    if doubt == "None" or doubt == "Unsure":
        # in this case, continue normal conversation
        #print("Doubt status: ", doubt)
        if doubt == "None":
            context.append({'role':'system', 'content':"""No doubt was found. Ask again if the user has any doubts on electric circuits."""})
        if doubt == "Unsure":
            context.append({'role':'system', 'content':"""The doubt was unclear. Ask the user to rephrase it better."""})
        response = get_completion_from_messages(context)
    elif doubt == "balance_depleted":
        response = "Estou em greve. Retornarei ao trabalho assim que as condições na OpenAI forem normalizadas."
    else: 
        # in this case, search for a lesson recommendation
        #print(doubt)
        present_keywords = matching_keywords_with_modules(doubt)
        modules, max_len = select_modules_for_search(present_keywords)
        if len(modules) > 0:
            if len(modules) < 4 and max_len > 1:
                response_as_recommendation = True
                # First, a quick message to say the assistant is searching the course for a lesson
                bot.reply_to(message, "Espere um pouco que estou procurando uma aula para te indicar.")
                for module in modules:
                    filename = module_index_to_filename(module)
                    recommended_lesson, duration = recommendations_from_strings(filename, doubt)
                    link = recommeded_lesson_to_link(module, recommended_lesson)
                    response = recommend_lesson_dumb(recommended_lesson, duration, modules_names[module], link)
                    responses_List.append(response)
            else:
                context.append({'role':'system', 'content':"""It seems the subject of the doubt was too broad. Then asks the user to be more specific, using more words."""})
                response = get_completion_from_messages(context)
        else:
            context.append({'role':'system', 'content':"""It seems the student does not have a doubt. Then asks what the user wants."""})
            response = get_completion_from_messages(context)
    context.append({'role':'assistant', 'content':f"{response}"}) #Append only the last response generated if for loop was executed
    if response_as_recommendation == False:
        bot.reply_to(message, response)
    else:
        bot.reply_to(message, "Aqui estão algumas aulas que podem te ajudar:")
        for response in responses_List:
            bot.reply_to(message, response)
    #print("Context shape:", len(context))

# This function ensures infinite polling of messages until the program is shut down
bot.infinity_polling()
