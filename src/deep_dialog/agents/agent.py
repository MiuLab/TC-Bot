"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

from deep_dialog import dialog_config

class Agent:
    """ Prototype for all agent classes, defining the interface they must uphold """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        """ Constructor for the Agent class

        Arguments:
        movie_dict      --  This is here now but doesn't belong - the agent doesn't know about movies
        act_set         --  The set of acts. #### Shouldn't this be more abstract? Don't we want our agent to be more broadly usable?
        slot_set        --  The set of available slots
        """
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        self.current_action = {}                    #   TODO Changed this variable's name to current_action
        self.current_action['diaact'] = None        #   TODO Does it make sense to call it a state if it has an act? Which act? The Most recent?
        self.current_action['inform_slots'] = {}
        self.current_action['request_slots'] = {}
        self.current_action['turn'] = 0

    def state_to_action(self, state, available_actions):
        """ Take the current state and return an action according to the current exploration/exploitation policy

        We define the agents flexibly so that they can either operate on act_slot representations or act_slot_value representations.
        We also define the responses flexibly, returning a dictionary with keys [act_slot_response, act_slot_value_response]. This way the command-line agent can continue to operate with values

        Arguments:
        state      --   A tuple of (history, kb_results) where history is a sequence of previous actions and kb_results contains information on the number of results matching the current constraints.
        user_action         --   A legacy representation used to run the command line agent. We should remove this ASAP but not just yet
        available_actions   --   A list of the allowable actions in the current state

        Returns:
        act_slot_action         --   An action consisting of one act and >= 0 slots as well as which slots are informed vs requested.
        act_slot_value_action   --   An action consisting of acts slots and values in the legacy format. This can be used in the future for training agents that take value into account and interact directly with the database
        """
        act_slot_response = None
        act_slot_value_response = None
        return {"act_slot_response": act_slot_response, "act_slot_value_response": act_slot_value_response}


    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """  Register feedback from the environment, to be stored as future training data

        Arguments:
        s_t                 --  The state in which the last action was taken
        a_t                 --  The previous agent action
        reward              --  The reward received immediately following the action
        s_tplus1            --  The state transition following the latest action
        episode_over        --  A boolean value representing whether the this is the final action.

        Returns:
        None
        """
        pass
    
    
    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model  
    
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model
     
       
    def add_nl_to_action(self, agent_action):
        """ Add NL to Agent Dia_Act """
        
        if agent_action['act_slot_response']:
            agent_action['act_slot_response']['nl'] = ""
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_response'], 'agt') #self.nlg_model.translate_diaact(agent_action['act_slot_response']) # NLG
            agent_action['act_slot_response']['nl'] = user_nlg_sentence
        elif agent_action['act_slot_value_response']:
            agent_action['act_slot_value_response']['nl'] = ""
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_value_response'], 'agt') #self.nlg_model.translate_diaact(agent_action['act_slot_value_response']) # NLG
            agent_action['act_slot_response']['nl'] = user_nlg_sentence