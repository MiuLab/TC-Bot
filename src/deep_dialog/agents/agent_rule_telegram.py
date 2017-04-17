from .agent import Agent
from deep_dialog import dialog_config
import numpy as np


class RuleAgent(Agent):
    """ A simple agent that informs all requested slots,
    then issues inform(taskcomplete) when the user stops making requests. """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """

        ########################################################################
        # find out if the user is requesting anything
        # if so, inform it
        # if he doesn't request smth
        ########################################################################

        print(">>> RuleAgent, STATE: ", state)
        user_action = state['user_action']
        self.state['turn'] += 2
        act_slot_response = {}
        act_slot_response['inform_slots'] = {}
        act_slot_response['request_slots'] = {}

        if user_action['diaact'] == 'request':
            requested_slot = list(user_action['request_slots'].keys())[0]
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'][requested_slot] = "PLACEHOLDER"

        elif user_action['diaact'] == 'inform':
            act_slot_response['diaact'] = "request"
            requestable_slots = list(set(dialog_config.sys_ask_slots) ^
                                     set(state['current_slots']['inform_slots'].keys()) ^
                                     set(state['current_slots']['proposed_slots'].keys()))
            act_slot_response['request_slots'] = {np.random.choice(requestable_slots): "UNK"}
        else:
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots']['moviename'] = "PLACEHOLDER"

        act_slot_response['turn'] = self.state['turn']

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
