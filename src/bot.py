# -*- coding: utf-8 -*-

import config
import telebot
from emoji import emojize
import random
import pickle
import json
from deep_dialog.agents import AgentDQN
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims.realUser import RealUser
from telegramDialogClasses import TelegramDialogManager
from deep_dialog.dialog_system.dict_reader import text_to_dict


def load_file(file_name):
    try:
        with open(file_name, 'rb') as f:
            obj = pickle.load(f, encoding='latin1')
    except (UnicodeDecodeError, pickle.UnpicklingError):
        with open(file_name, "rt") as f:
            obj = json.load(f)
    return obj

######### --****-- Begin of initialization ---****-- ###########

bot = telebot.TeleBot(config.token)
turn_count = 0

movie_kb = load_file(config.movie_kb_path)
act_set = text_to_dict('./deep_dialog/data/dia_acts.txt')
slot_set = text_to_dict('./deep_dialog/data/slot_set.txt')
# movie_dictionary = load_file(config.dict_path)

nlg_model = nlg()
nlg_model.load_nlg_model(config.nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(config.diaact_nl_pairs)

nlu_model = nlu()
nlu_model.load_nlu_model(config.nlu_model_path)

agent = AgentDQN(movie_kb, act_set, slot_set, config.agent_params)
user = RealUser()

agent.set_nlg_model(nlg_model)
user.set_nlg_model(nlg_model)

agent.set_nlu_model(nlu_model)
user.set_nlu_model(nlu_model)

dia_manager = TelegramDialogManager(agent, user, act_set, slot_set, movie_kb)

######### --****-- End of initialization ---****-- ##############



@bot.message_handler(commands=['help'])
def handle_help(message):
    help_message = "Hello, friend!\n" + get_random_emoji(4) + \
                   "I can help you to buy tickets  " + emojize(":ticket:")+" to the cinema.\n" \
                   "=====================================\n" \
                   "* Print /start to start a conversation;\n" \
                   "* Print /end to end the dialog;\n"

    bot.send_message(message.chat.id, help_message)


@bot.message_handler(commands=['start'])
def handle_start(message):
    global turn_count
    turn_count = 1
    greetings = "Hello! I can help you to buy tickets to the cinema.\nWhat film would you like to watch?"
    bot.send_message(message.chat.id, greetings)


@bot.message_handler(commands=['end'])
def handle_end(message):
    global turn_count
    turn_count = 0
    goodbye = "Farewell! Let me know if you would like to buy tickets again." + get_random_emoji()
    bot.send_message(message.chat.id, goodbye)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    global turn_count
    if turn_count > 0:
        if turn_count == 1:
            dia_manager.initialize_episode(message.text)

        ## if turn_count >= 1:
        episode_over, agent_ans = dia_manager.next_turn(message.text)
        turn_count+=1
        bot.send_message(message.chat.id, agent_ans+' ' + get_random_emoji(1))
        if episode_over:
            turn_count = 0


def get_random_emoji(num = 1):
    emoji_list = [":rainbow:", ":octopus:", ":panda_face:",
                  ":sunny:", ":hibiscus:", ":rose:", ":whale:",
                  ":full_moon_with_face:", ":earth_americas:",
                  ":hatching_chick:", ":video_camera:", ":tv:", ":ghost:",
                  ":sunrise:", ":city_sunrise:", ":stars:", ":ticket:", ":moyai:"]
    res_list = []
    for i in range(num):
        randnum = random.randint(0, len(emoji_list)-1)
        res_list.append(emojize(emoji_list[randnum], use_aliases=True))

    return "".join(res_list) + '\n'


if __name__ == '__main__':
    bot.polling(none_stop=True)
