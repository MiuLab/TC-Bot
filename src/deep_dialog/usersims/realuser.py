"""
Created on October 5, 2017

a real user simulator

@author: yuta-kawai
"""

from .usersim import UserSimulator

from deep_dialog import dialog_config


class RealUser(UserSimulator):
    def __init__(self, movie_dict, act_set, slot_set, start_set, params):
        """ Constructor shared by all user simulators """
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        self.simulator_act_level = params['simulator_act_level']
        self.learning_phase = params['learning_phase']
        self.goal = {}

    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        # first action
        user_action = self._sample_action()

        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def _sample_action(self):
        """ sample a start action """
        user_action = self.generate_diaact_from_nl(raw_input('> '))
        user_action['turn'] = self.state['turn']

        self.state['diaact'] = user_action['diaact']
        self.state['inform_slots'] = user_action['inform_slots']
        self.state['request_slots'] = user_action['request_slots']

        return user_action

    def next(self, system_action):
        """ Generate next User Action based on last System Action """

        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        sys_act = system_action['diaact']

        response_action = {}
        if sys_act == 'thanks':
            # dummy
            response_action = self.generate_diaact_from_nl('')
        else:
            response_action = self.generate_diaact_from_nl(raw_input('> '))

        response_action['turn'] = self.state['turn']

        self.state['diaact'] = response_action['diaact']
        self.state['inform_slots'] = response_action['inform_slots']
        self.state['request_slots'] = response_action['request_slots']

        if (self.max_turn > 0) and (self.state['turn'] > self.max_turn):
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            if sys_act == "closing":
                self.episode_over = True
                self.state['diaact'] = "thanks"
            elif sys_act == "thanks":
                self.episode_over = True
                self.dialog_status = dialog_config.SUCCESS_DIALOG

        response_action['diaact'] = self.state['diaact']

        return response_action, self.episode_over, self.dialog_status

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
