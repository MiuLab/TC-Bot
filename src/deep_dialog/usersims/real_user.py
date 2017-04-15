"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

"Suitable only for message given"

from .usersim import UserSimulator


class RealUser(UserSimulator):

    def __init__(self):
        """ Constructor for the Agent class """
        pass


    def state_to_action(self, message):
        """ Generate an action by getting input interactively from the command line """
        act_slot_value_response = self.generate_diaact_from_nl(message)

        return {"act_slot_response": act_slot_value_response, "act_slot_value_response": act_slot_value_response}


    def next(self, message):

        user_action = self.generate_diaact_from_nl(message)
        return user_action


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
