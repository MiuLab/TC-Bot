import copy


class StateTracker:
    """ The state tracker maintains a record of which request slots are filled and which inform slots are filled """

    def __init__(self, kb_helper):
        """ constructor for statetracker takes kb_helper and initializes a new episode

        Class Variables:
        
        history_dictionaries    --  A record of the current dialog in dictionary format
        current_slots           --  A dictionary that keeps a running record of which slots are filled current_slots['inform_slots']
                                    and which are requested current_slots['request_slots'] (but not filed)
        turn_count              --  A running count of which turn we are at in the present dialog
        
        """
        self.initialize_episode()
        self.history_dictionaries = None
        self.current_slots = None
        self.turn_count = 0
        self.kb_helper = kb_helper

    def initialize_episode(self):
        """ Initialize a new episode (dialog), 
        flush the current state and tracked slots """

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

    def dialog_history_dictionaries(self):

        """  Return the dictionary representation 
        of the dialog history (includes values) """

        return self.history_dictionaries

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

    def get_state_for_agent(self):
        """ Get the state representatons to send to agent """
        state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots,
                 'kb_results_dict': self.kb_helper.database_results_for_agent(self.current_slots),
                 'turn': self.turn_count, 'history': self.history_dictionaries,
                 'agent_action': self.history_dictionaries[-2] if len(self.history_dictionaries) > 1 else None}
        return state

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

        print('StateTracker, current_slots: ', self.current_slots)
        ########################################################################
        #   Update state to reflect a new action by the agent.
        ########################################################################
        if agent_action:

            ####################################################################
            #   Handles the act_slot response (with values needing to be filled)
            ####################################################################
            # if agent_action['act_slot_response']:
            response = copy.deepcopy(agent_action.get('act_slot_response', None))
            assert response  ## raise exception if response is None

            inform_slots = self.kb_helper.fill_inform_slots(response['inform_slots'], self.current_slots)
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

            user_action_values = {'turn': self.turn_count, 'speaker': "user",
                                  'request_slots': user_action['request_slots'],
                                  'inform_slots': user_action['inform_slots'], 'diaact': user_action['diaact']}

            self.history_dictionaries.append(user_action_values)

        self.turn_count += 1
