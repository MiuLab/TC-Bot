"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

"Suitable only for message given"

from .usersim import UserSimulator
from .. import dialog_config



class RealUser(UserSimulator):
    def __init__(self, act_set=None, slot_set=None, params=None):
        """ Constructor for the Agent class """

        # self.movie_dict = movie_dict
        # self.act_set = act_set
        # self.slot_set = slot_set
        # self.act_cardinality = len(list(act_set.keys()))
        # self.slot_cardinality = len(list(slot_set.keys()))
        # self.agent_run_mode = params['agent_run_mode']
        # self.agent_act_level = params['agent_act_level']

    def state_to_action(self, state, message):
        """ Generate an action by getting input interactively from the command line """
        act_slot_value_response = self.generate_diaact_from_nl(message)
        return {"act_slot_response": act_slot_value_response, "act_slot_value_response": act_slot_value_response}

    # def initialize_episode(self, message):
    #     """ Initialize a new episode (dialog)"""
    #
    #     """ Initialize a new episode. This function is called every time a new episode is run. """
    #
    #     user_action = self.generate_diaact_from_nl(message)
    #     # self.goal = random.choice(self.start_set)
    #     # self.goal['request_slots']['ticket'] = 'UNK'
    #     # episode_over, user_action = self._sample_action()
    #     # assert (episode_over != 1), ' but we just started'
    #
    #     return user_action

    def next(self, message):

        user_action = self.generate_diaact_from_nl(message)
        # episode_over = False
        # dialog_status = dialog_config.NO_OUTCOME_YET

        # add NL to dia_act
        # return response_action, self.episode_over, self.dialog_status

        return user_action #, episode_over, dialog_status


    def generate_diaact_from_nl(self, string):
        """ Generate Dia_Act Form with NLU """

        user_action = {}
        user_action['diaact'] = 'UNK'
        user_action['inform_slots'] = {}
        user_action['request_slots'] = {}

        if len(string) > 0:
            user_action = self.nlu_model.generate_dia_act(string)

        user_action['nl'] = string
        return user_action
