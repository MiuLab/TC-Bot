# -*- coding: utf-8 -*-


token = '358288982:AAGTQa1xSExTiBHOJa3BriAH7sp2sMSalrc'
trained_model_path = './deep_dialog/checkpoints/rl_agent/right_model.p'

nlg_model_path = './deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.json'
nlu_model_path = './deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.json'

diaact_nl_pairs = './deep_dialog/data/dia_act_nl_pairs.v6.json'
movie_kb_path = './deep_dialog/data/movie_kb.1k.json'
# dict_path = './deep_dialog/data/dicts.v3.json'

agent_params = {'max_turn': 40,
                'epsilon': 0,         # Epsilon to determine stochasticity of epsilon-greedy agent policies
                'agent_run_mode': 0,  # For NL
                'agent_act_level': 1, ## TODO: maybe throw it away???
                'experience_replay_pool_size': 1000,
                'dqn_hidden_size': 60,
                'batch_size': 16,
                'gamma': 0.9,
                'predict_mode': True,
                'trained_model_path': trained_model_path,
                'warm_start': 1,
                'cmd_input_mode': 0} ## TODO: maybe throw it away???


