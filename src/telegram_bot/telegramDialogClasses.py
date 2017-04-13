# from deep_dialog.dialog_system.state_tracker import StateTracker
from collections import Counter

from fuzzywuzzy import fuzz

from src.deep_dialog import dialog_config
from src.deep_dialog.agents.agent import Agent
# from deep_dialog.dialog_system.kb_helper import KBHelper
# from collections import defaultdict
import numpy as np
import copy


class TelegramDialogManager():
    def __init__(self, agent, user, act_set, slot_set, movie_dictionary):
        self.agent = agent
        self.user = user
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = TelegramStateTracker(act_set, slot_set, movie_dictionary)
        self.episode_over = False

    def initialize_episode(self):
        """ Refresh state for new dialog """

        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.agent.initialize_episode()

    def next_turn(self, message):
        """ This function initiates each subsequent exchange between agent and user (agent first) """

        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################

        user_action = self.user.next(message)
        self.state_tracker.update(user_action=user_action)

        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################

        agent_state = self.state_tracker.get_state_for_agent()
        agent_action = self.agent.state_to_action(agent_state)

        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=agent_action)
        print("==== next_turn, agent_action: ", agent_action)

        agent_ans = self.agent.add_nl_to_action(agent_action)

        if user_action['diaact'] == "thanks":
            agent_ans = 'Thank you, good bye!'
            self.episode_over = True

        ########################################################################
        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
        ########################################################################
        return self.episode_over, agent_ans


class TelegramStateTracker:
    """ The state tracker maintains a record of which request slots are filled and which inform slots are filled """

    def __init__(self, act_set, slot_set, movie_dictionary):
        """ constructor for statetracker takes movie knowledge base and initializes a new episode

        Arguments:
        act_set                 --  The set of all acts availavle
        slot_set                --  The total set of available slots
        movie_dictionary        --  A representation of all the available movies. Generally this object is accessed via the KBHelper class

        Class Variables:
        history_vectors         --  A record of the current dialog so far in vector format (act-slot, but no values)
        history_dictionaries    --  A record of the current dialog in dictionary format
        current_slots           --  A dictionary that keeps a running record of which slots are filled current_slots['inform_slots'] and which are requested current_slots['request_slots'] (but not filed)
        action_dimension        --  # TODO indicates the dimensionality of the vector representaiton of the action
        kb_result_dimension     --  A single integer denoting the dimension of the kb_results features.
        turn_count              --  A running count of which turn we are at in the present dialog
        """
        # self.movie_dictionary = movie_dictionary
        self.initialize_episode()
        self.history_vectors = None
        self.history_dictionaries = None
        self.current_slots = None
        self.turn_count = 0
        self.kb_helper = TelegramKBHelper(movie_dictionary)
        # self.kb_helper = KBHelper(movie_dictionary)


    def initialize_episode(self):
        """ Initialize a new episode (dialog), flush the current state and tracked slots """

        self.history_dictionaries = []
        self.turn_count = 0
        self.current_slots = {}

        """
            'inform_slots'          --  which slots is became known on the current turn from user;
            'request_slots'         --  which slots user requested on the current turn;
            'proposed_slots'        --  all known from user and agent inform slots from the begging of the session;
            'agent_request_slots'   --  what slots agent would like to request on the current turn;

        """
        self.current_slots['inform_slots'] = {}
        self.current_slots['request_slots'] = {}
        self.current_slots['proposed_slots'] = {}
        self.current_slots['agent_request_slots'] = {}

    def get_state_for_agent(self):
        """ Get the state representatons to send to agent """

        state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots,
                 'kb_results_dict': self.kb_helper.database_results_for_agent(self.current_slots),
                 'turn': self.turn_count, 'history': self.history_dictionaries,
                 'agent_action': self.history_dictionaries[-2] if len(self.history_dictionaries) > 1 else None}
        return state ### that was returned with copy.deepcopy before my corrections;


    def update(self, agent_action=None, user_action=None):

        """
            Update the state based on the latest action;
             The first goal of the method -- adding agent_action or user_action
             to the self.history_dictionaries;
             The last one is necessary for 'get_state_for_agent' function;

             The second goal of the method -- updating self.current_slots;

        """

        ########################################################################
        #  Make sure that the function was called properly
        ########################################################################
        assert (not (user_action and agent_action))
        assert (user_action or agent_action)

        ########################################################################
        #   Update state to reflect a new action by the agent.
        ########################################################################
        if agent_action:

            ####################################################################
            #   Handles the act_slot response (with values needing to be filled)
            ####################################################################
            # if agent_action['act_slot_response']:
            response = copy.deepcopy(agent_action.get('act_slot_response', None))
            assert response ## raise exception if response is None

            inform_slots = self.kb_helper.fill_inform_slots(response['inform_slots'], self.current_slots)
            # TODO this doesn't actually work yet, remove this warning when kb_helper is functional
            agent_action_values = {'turn': self.turn_count, 'speaker': "agent", 'diaact': response['diaact'],
                                   'inform_slots': inform_slots, 'request_slots': response['request_slots']}

            agent_action['act_slot_response'].update({'diaact': response['diaact'], 'inform_slots': inform_slots,
                                                      'request_slots': response['request_slots'],
                                                      'turn': self.turn_count})


            ### delete from self.current_slots['request_slots'][slot] slot which user requested;
            ### and also updating request/inform/propose slots;
            for slot in agent_action_values['inform_slots'].keys():
                self.current_slots['proposed_slots'][slot] = agent_action_values['inform_slots'][slot]
                self.current_slots['inform_slots'][slot] = agent_action_values['inform_slots'][slot]
                if slot in self.current_slots['request_slots'].keys():
                    del self.current_slots['request_slots'][slot]

            #### add slots, which agent requests to the self.current_slots['agent_request_slots'][slot]
            for slot in agent_action_values['request_slots'].keys():
                if slot not in self.current_slots['agent_request_slots']:
                    self.current_slots['agent_request_slots'][slot] = "UNK"

            self.history_dictionaries.append(agent_action_values)

        ########################################################################
        #   Update the state to reflect a new action by the user
        ########################################################################
        elif user_action:

            ####################################################################
            #   Update the current slots
            ####################################################################

            for slot in user_action['inform_slots'].keys():
                self.current_slots['inform_slots'][slot] = user_action['inform_slots'][slot]
                if slot in self.current_slots['request_slots'].keys():
                    del self.current_slots['request_slots'][slot]

            for slot in user_action['request_slots'].keys():
                if slot not in self.current_slots['request_slots']:
                    self.current_slots['request_slots'][slot] = "UNK"

            user_action_values = {'turn': self.turn_count, 'speaker': "user", 'request_slots': user_action['request_slots'],
                        'inform_slots': user_action['inform_slots'], 'diaact': user_action['diaact']}

            self.history_dictionaries.append(user_action_values)

        self.turn_count += 1
        print("====== State_tracker: turn{}, current_slots: {}".format(self.turn_count, self.current_slots))


class TelegramKBHelper:
    """ An assistant to fill in values for the agent (which knows about slots of values) """

    def __init__(self, movie_dictionary, cmp_limit=90):
        """ Constructor for a KBHelper """

        self.cmp_limit = cmp_limit
        self.movie_dictionary = movie_dictionary
        self.cached_kb_slot = {}

    def fill_inform_slots(self, inform_slots_to_be_filled, current_slots):
        """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)

        Arguments:
        inform_slots_to_be_filled   --  Something that looks like {starttime:None, theater:None} where starttime and theater are slots that the agent needs filled
        current_slots               --  Contains a record of all filled slots in the conversation so far - for now, just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots

        Returns:
        filled_slots             --  A dictionary of form {slot1:value1, slot2:value2} for each sloti in inform_slots_to_be_filled
        """
        kb_results = self.available_results_from_kb(current_slots)
        current_slots = current_slots['inform_slots'] ## just already filled slots

        filled_slots = {}
        if 'taskcomplete' in inform_slots_to_be_filled.keys():
            filled_slots.update(current_slots) ## TODO: check to be sure if it's ok

        #### TODO: very strange part of code begins here:

        for slot in inform_slots_to_be_filled.keys():
            if slot == 'numberofpeople':
                if slot in current_slots.keys():
                    filled_slots[slot] = current_slots[slot]
                else:
                    filled_slots[slot] = 'a lot of'  ## very bad to use this trick, but... :(

            if slot == 'ticket' or slot == 'taskcomplete':
                filled_slots[slot] = dialog_config.TICKET_AVAILABLE if len(
                    kb_results) > 0 else dialog_config.NO_VALUE_MATCH
                continue

            if slot == 'closing':
                continue

            ####################################################################
            #   Grab the value for the slot with the highest count and fill it
            ####################################################################

            values_counts = Counter(self.available_slot_values(slot, kb_results).values()).items()
            if len(values_counts) > 0:
                filled_slots[slot] = sorted(values_counts, key=lambda x: -x[1])[0][0]
            else:
                filled_slots[slot] = dialog_config.NO_VALUE_MATCH  # "NO VALUE MATCHES SNAFU!!!"

        return filled_slots

    def available_slot_values(self, slot, kb_results):
        """ Return the set of values available for the slot based on the current constraints """

        slot_values = {}
        for movie_id in kb_results.keys():
            if slot in kb_results[movie_id].keys():
                slot_val = kb_results[movie_id][slot]
                if slot_val in slot_values.keys():
                    slot_values[slot_val] += 1
                else:
                    slot_values[slot_val] = 1
        return slot_values

    def available_results_from_kb(self, current_slots):

        ret_result = []  ## what we should return;
        current_slots = current_slots['inform_slots']

        if len(current_slots) == 0:
            print("HERE!1")
            return self.movie_dictionary

        constrain_keys = current_slots.keys()

        ### leave only the keys which could be constraits ###
        constrain_keys = list(set(constrain_keys) & set(dialog_config.sys_inform_slots))
        constrain_keys = [k for k in constrain_keys if current_slots[k] != dialog_config.I_DO_NOT_CARE]

        if len(constrain_keys) == 0:
            print("HERE!2")
            return self.movie_dictionary

        ### starting looking for the suitable records:
        for id_ in self.movie_dictionary.keys():
            kb_keys = self.movie_dictionary[id_].keys()

            ### checking if all constraits is available
            ### for the current film id:

            if set(constrain_keys).issubset(set(kb_keys)):
                # print('HERE 3')
                ### check if all constraints are the same in current_slots and in movie_dict:
                ### for that current film id:
                match = True
                for k in constrain_keys:
                    if fuzz.ratio(str(current_slots[k]), str(self.movie_dictionary[id_][k])) > self.cmp_limit:
                        continue
                    else:
                        match = False
                        break
                if match:
                    ret_result.append((id_, self.movie_dictionary[id_]))

        ### add everything to cache;
        query_idx_keys = frozenset(current_slots.items())
        self.cached_kb_slot.update({query_idx_keys:ret_result})

        ret_result = dict(ret_result) ## It was dict in the previous version;
        print("--------available_results_from_kb: ", ret_result)
        return ret_result

    def database_results_for_agent(self, current_slots):
        """ A dictionary of the number of results matching each current constraint.
        The agent needs this to decide what to do next.
        Return the count statistics for each constraint in inform_slots """

        # print("----------database_results_for_agent (cur_slots): ",current_slots)
        inform_slots = current_slots['inform_slots']
        kb_results = {key: 0 for key in inform_slots.keys()}
        kb_results['matching_all_constraints'] = 0

        query_idx_keys = frozenset(inform_slots.items())
        cached_kb_slot_ret = self.cached_kb_slot.get(query_idx_keys, [])

        if len(cached_kb_slot_ret) > 0:
            return cached_kb_slot_ret

        for movie_id in self.movie_dictionary.keys():
            all_slots_match = 1
            for slot in inform_slots.keys():
                if slot == 'ticket' or inform_slots[slot] == dialog_config.I_DO_NOT_CARE:
                    continue

                if slot in self.movie_dictionary[movie_id]:
                    if fuzz.ratio(inform_slots[slot], self.movie_dictionary[movie_id][slot]) > self.cmp_limit:
                        kb_results[slot] += 1
                    else:
                        all_slots_match = 0
                else:
                    all_slots_match = 0
            kb_results['matching_all_constraints'] += all_slots_match

        self.cached_kb_slot.update({query_idx_keys:kb_results})
        # database_results = self.available_results_from_kb_for_slots(current_slots['inform_slots'])
        # print("----------database_results_for_agent: ", kb_results)
        return kb_results

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

        print("*******************STATE: ",state)
        user_action = state['user_action']
        self.state['turn'] += 2
        act_slot_response = {}
        act_slot_response['inform_slots'] = {}
        act_slot_response['request_slots'] = {}

        ########################################################################
        # find out if the user is requesting anything
        # if so, inform it
        ########################################################################

        if user_action['diaact'] == 'request':
            requested_slot = list(user_action['request_slots'].keys())[0]
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'][requested_slot] = "PLACEHOLDER"

        if user_action['diaact'] == 'inform':
            act_slot_response['diaact'] = "request"
            requestable_slots = list(set(dialog_config.sys_ask_slots)^
                                     set(state['current_slots']['inform_slots'].keys())^
                                     set(state['current_slots']['proposed_slots'].keys()))
            act_slot_response['request_slots'] = {np.random.choice(requestable_slots):"UNK"}
        else:
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots']['moviename'] = "PLACEHOLDER"

        act_slot_response['turn'] = self.state['turn']

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
