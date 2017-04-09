# -*- coding: utf-8 -*-

import telebot
from emoji import emojize
import random
import pickle
import json
# from deep_dialog.agents import AgentDQN
# from deep_dialog.agents import agent_baselines
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims.realUser import RealUser
from telegramDialogClasses import TelegramDialogManager
from telegramDialogClasses import RuleAgent
from deep_dialog.dialog_system.dict_reader import text_to_dict
import telebot
import cherrypy


def load_file(file_name):
    try:
        with open(file_name, 'rb') as f:
            obj = pickle.load(f, encoding='latin1')
    except (UnicodeDecodeError, pickle.UnpicklingError):
        with open(file_name, "rt") as f:
            obj = json.load(f)
    return obj


######### --****-- Begin of initialization ---****-- ###########

WEBHOOKS_AVAIL = False

config = load_file("/Users/fogside/Projects/Telegram_bot/RL-Dialog-Bot/src/telegram_bot/config.json")
bot = telebot.TeleBot(config["token"])

turn_count = 0
movie_kb = load_file(config["movie_kb_path"])
act_set = text_to_dict(config['act_set_path'])
slot_set = text_to_dict(config['slot_set_path'])

nlg_model = nlg()
nlg_model.load_nlg_model(config['nlg_model_path'])
nlg_model.load_predefine_act_nl_pairs(config['diaact_nl_pairs'])

nlu_model = nlu()
nlu_model.load_nlu_model(config['nlu_model_path'])

# agent = AgentDQN(movie_kb, act_set, slot_set, config['agent_params'])
# agent = agent_baselines.EchoAgent(movie_kb, act_set, slot_set, config['agent_params'])
agent = RuleAgent(movie_kb, act_set, slot_set, config['agent_params'])
user = RealUser()

agent.set_nlg_model(nlg_model)
# user.set_nlg_model(nlg_model)
# agent.set_nlu_model(nlu_model)
user.set_nlu_model(nlu_model)

dia_manager = TelegramDialogManager(agent, user, act_set, slot_set, movie_kb)

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

#########  --****---    End of initialization ---****--   ###########
######### --****--- Next code is for debugging ---****-- ############

#
# dia_manager.initialize_episode()
# turn_count+=1
#
# print("Hello! I can help you to buy tickets to the cinema.\nWhat film would you like to watch?")
#
# while(turn_count > 0):
#     msg = input()
#     episode_over, agent_ans = dia_manager.next_turn(msg)
#     turn_count+=1
#     # bot.send_message(msg, agent_ans+' ' + get_random_emoji(1))
#     print("turn #{}: {}".format(turn_count, agent_ans))
#     if episode_over:
#         turn_count = 0
#     if msg == 'stop':
#         turn_count = 0
#
# exit(0)
#


####### --** End of Debugging **-- ########



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

@bot.message_handler(commands=['films'])
def show_films(message):
    '''

    These should return a
    list of available films for user

    '''

    available_films = []
    warning = 'currently not available'
    if len(available_films) == 0:
        bot.send_message(message.chat.id, warning)
    else:
        bot.send_message(message.chat.id, available_films)


@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_text(message):
    global turn_count
    if turn_count > 0:
        if turn_count == 1:
            dia_manager.initialize_episode()

        episode_over, agent_ans = dia_manager.next_turn(message.text)
        turn_count+=1
        bot.send_message(message.chat.id, agent_ans+' ' + get_random_emoji(1))
        if episode_over:
            turn_count = 0
    else:
        bot.reply_to(message, message.text)


if WEBHOOKS_AVAIL:

    WEBHOOK_HOST = config['WEBHOOK_HOST']
    WEBHOOK_PORT = config['WEBHOOK_PORT']
    WEBHOOK_LISTEN = config['WEBHOOK_LISTEN']

    WEBHOOK_SSL_CERT = config['WEBHOOK_SSL_CERT']  ## sertificat path
    WEBHOOK_SSL_PRIV = config['WEBHOOK_SSL_PRIV']  ## private key path

    WEBHOOK_URL_BASE = "https://%s:%s" % (WEBHOOK_HOST, WEBHOOK_PORT)
    WEBHOOK_URL_PATH = "/%s/" % config['token']


    class WebhookServer(object):
        @cherrypy.expose
        def index(self):
            if 'content-length' in cherrypy.request.headers and \
                            'content-type' in cherrypy.request.headers and \
                            cherrypy.request.headers['content-type'] == 'application/json':
                length = int(cherrypy.request.headers['content-length'])
                json_string = cherrypy.request.body.read(length).decode("utf-8")
                update = telebot.types.Update.de_json(json_string)
                bot.process_new_updates([update])
                return ''
            else:
                raise cherrypy.HTTPError(403)


    # Снимаем вебхук перед повторной установкой (избавляет от некоторых проблем)
    bot.remove_webhook()

    # Ставим заново вебхук
    bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH,
                    certificate=open(WEBHOOK_SSL_CERT, 'r'))

    # Указываем настройки сервера CherryPy
    cherrypy.config.update({
        'server.socket_host': WEBHOOK_LISTEN,
        'server.socket_port': WEBHOOK_PORT,
        'server.ssl_module': 'builtin',
        'server.ssl_certificate': WEBHOOK_SSL_CERT,
        'server.ssl_private_key': WEBHOOK_SSL_PRIV
    })


    cherrypy.quickstart(WebhookServer(), WEBHOOK_URL_PATH, {'/': {}})

else:
    bot.delete_webhook()
    bot.polling(none_stop=True) ## uncomment it for local testing;
