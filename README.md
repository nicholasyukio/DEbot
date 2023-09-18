# DEbot
OpenAI based Telegram bot that recommends lessons based on user's input messages

It uses the OpenAI API and the Telegram API.

This bot was created for students of the online course Domínio Elétrico, which is about the subject of electric circuits.

The keywords.json file contain keywords for each module of the course, and the other JSON files (modulo_XX.json and DE_Labs.json) contain the title and duration of each lesson.

This bot uses the OpenAI API so it can engage in conversations with the user inteligently, and whenever the uses presents a doubt or difficulty studying the subject of the course, the bot summarizes the doubt topic in a few words, compare the words obtained with the keywords listed in keywords.json file, selects some modules which might be related to the doubt, and then proceeds to look for some lesson recommendation for the user.
