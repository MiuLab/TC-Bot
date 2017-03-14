"""
Created on May 20, 2016

state tracker

@author: xiul, t-zalipt
"""

from . import KBHelper
import numpy as np
import copy


class StateTracker:
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
        self.movie_dictionary = movie_dictionary
        self.initialize_episode()
        self.history_vectors = None
        self.history_dictionaries = None
        self.current_slots = None
        self.action_dimension = 10      # TODO REPLACE WITH REAL VALUE
        self.kb_result_dimension = 10   # TODO  REPLACE WITH REAL VALUE
        self.turn_count = 0
        self.kb_helper = KBHelper(movie_dictionary)
        

    def initialize_episode(self):
        """ Initialize a new episode (dialog), flush the current state and tracked slots """
        
        self.action_dimension = 10
        self.history_vectors = np.zeros((1, self.action_dimension))
        self.history_dictionaries = []
        self.turn_count = 0
        self.current_slots = {}
        
        self.current_slots['inform_slots'] = {}
        self.current_slots['request_slots'] = {}
        self.current_slots['proposed_slots'] = {}
        self.current_slots['agent_request_slots'] = {}


    def dialog_history_vectors(self):
        """ Return the dialog history (both user and agent actions) in vector representation """
        return self.history_vectors


    def dialog_history_dictionaries(self):
        """  Return the dictionary representation of the dialog history (includes values) """
        return self.history_dictionaries


    def kb_results_for_state(self):
        """ Return the information about the database results based on the currently informed slots """
        ########################################################################
        # TODO Calculate results based on current informed slots
        ########################################################################
        kb_results = self.kb_helper.database_results_for_agent(self.current_slots) # replace this with something less ridiculous
        # TODO turn results into vector (from dictionary)
        results = np.zeros((0, self.kb_result_dimension))
        return results
        

    def get_state_for_agent(self):
        """ Get the state representatons to send to agent """
        #state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots, 'kb_results': self.kb_results_for_state()}
        state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots, #'kb_results': self.kb_results_for_state(), 
                 'kb_results_dict':self.kb_helper.database_results_for_agent(self.current_slots), 'turn': self.turn_count, 'history': self.history_dictionaries, 
                 'agent_action': self.history_dictionaries[-2] if len(self.history_dictionaries) > 1 else None}
        return copy.deepcopy(state)
    
    def get_suggest_slots_values(self, request_slots):
        """ Get the suggested values for request slots """
        
        suggest_slot_vals = {}
        if len(request_slots) > 0: 
            suggest_slot_vals = self.kb_helper.suggest_slot_values(request_slots, self.current_slots)
        
        return suggest_slot_vals
    
    def get_current_kb_results(self):
        """ get the kb_results for current state """
        kb_results = self.kb_helper.available_results_from_kb(self.current_slots)
        return kb_results
    
    
    def update(self, agent_action=None, user_action=None):
        """ Update the state based on the latest action """

        ########################################################################
        #  Make sure that the function was called properly
        ########################################################################
        assert(not (user_action and agent_action))
        assert(user_action or agent_action)

        ########################################################################
        #   Update state to reflect a new action by the agent.
        ########################################################################
        if agent_action:
            
            ####################################################################
            #   Handles the act_slot response (with values needing to be filled)
            ####################################################################
            if agent_action['act_slot_response']:
                response = copy.deepcopy(agent_action['act_slot_response'])
                
                inform_slots = self.kb_helper.fill_inform_slots(response['inform_slots'], self.current_slots) # TODO this doesn't actually work yet, remove this warning when kb_helper is functional
                agent_action_values = {'turn': self.turn_count, 'speaker': "agent", 'diaact': response['diaact'], 'inform_slots': inform_slots, 'request_slots':response['request_slots']}
                
                agent_action['act_slot_response'].update({'diaact': response['diaact'], 'inform_slots': inform_slots, 'request_slots':response['request_slots'], 'turn':self.turn_count})
                
            elif agent_action['act_slot_value_response']:
                agent_action_values = copy.deepcopy(agent_action['act_slot_value_response'])
                # print("Updating state based on act_slot_value action from agent")
                agent_action_values['turn'] = self.turn_count
                agent_action_values['speaker'] = "agent"
                
            ####################################################################
            #   This code should execute regardless of which kind of agent produced action
            ####################################################################
            for slot in agent_action_values['inform_slots'].keys():
                self.current_slots['proposed_slots'][slot] = agent_action_values['inform_slots'][slot]
                self.current_slots['inform_slots'][slot] = agent_action_values['inform_slots'][slot] # add into inform_slots
                if slot in self.current_slots['request_slots'].keys():
                    del self.current_slots['request_slots'][slot]

            for slot in agent_action_values['request_slots'].keys():
                if slot not in self.current_slots['agent_request_slots']:
                    self.current_slots['agent_request_slots'][slot] = "UNK"

            self.history_dictionaries.append(agent_action_values)
            current_agent_vector = np.ones((1, self.action_dimension))
            self.history_vectors = np.vstack([self.history_vectors, current_agent_vector])
                            
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
            
            self.history_vectors = np.vstack([self.history_vectors, np.zeros((1,self.action_dimension))])
            new_move = {'turn': self.turn_count, 'speaker': "user", 'request_slots': user_action['request_slots'], 'inform_slots': user_action['inform_slots'], 'diaact': user_action['diaact']}
            self.history_dictionaries.append(copy.deepcopy(new_move))

        ########################################################################
        #   This should never happen if the asserts passed
        ########################################################################
        else:
            pass

        ########################################################################
        #   This code should execute after update code regardless of what kind of action (agent/user)
        ########################################################################
        self.turn_count += 1