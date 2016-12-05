"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""


from agent import Agent

class AgentCmd(Agent):
    
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        """ Constructor for the Agent class """
        
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.agent_input_mode = params['cmd_input_mode']
        
        
    def state_to_action(self, state):
        """ Generate an action by getting input interactively from the command line """

        user_action = state['user_action']
        # get input from the command line
        print "Turn", user_action['turn'] + 1, "sys:",
        command = raw_input()
        
        if self.agent_input_mode == 0: # nl
            act_slot_value_response = self.generate_diaact_from_nl(command)
        elif self.agent_input_mode == 1: # dia_act
            act_slot_value_response = self.parse_str_to_diaact(command)
        
        return {"act_slot_response": act_slot_value_response, "act_slot_value_response": act_slot_value_response}
    
    def parse_str_to_diaact(self, string):
        """ Parse string into Dia_Act Form """
        
        annot = string.strip(' ').strip('\n').strip('\r')
        act = annot

        if annot.find('(') > 0 and annot.find(')') > 0:
            act = annot[0: annot.find('(')].strip(' ').lower() #Dia act
            annot = annot[annot.find('(')+1:-1].strip(' ') #slot-value pairs
        else: annot = ''
        
        act_slot_value_response = {}
        act_slot_value_response['diaact'] = 'UNK'
        act_slot_value_response['inform_slots'] = {}
        act_slot_value_response['request_slots'] = {}
        
        if act in self.act_set: # dialog_config.all_acts
            act_slot_value_response['diaact'] = act
        else:
            print ("Something wrong for your input dialog act! Please check your input ...")

        if len(annot) > 0: # slot-pair values: slot[val] = id
            annot_segs = annot.split(';') #slot-value pairs
            sent_slot_vals = {} # slot-pair real value
            sent_rep_vals = {} # slot-pair id value

            for annot_seg in annot_segs:
                annot_seg = annot_seg.strip(' ')
                annot_slot = annot_seg
                if annot_seg.find('=') > 0:
                    annot_slot = annot_seg[:annot_seg.find('=')] 
                    annot_val = annot_seg[annot_seg.find('=')+1:]
                else: #requested
                    annot_val = 'UNK' # for request
                    if annot_slot == 'taskcomplete': annot_val = 'FINISH'

                if annot_slot == 'mc_list': continue

                # slot may have multiple values
                sent_slot_vals[annot_slot] = []
                sent_rep_vals[annot_slot] = []

                if annot_val.startswith('{') and annot_val.endswith('}'):
                    annot_val = annot_val[1:-1]

                    if annot_slot == 'result':
                        result_annot_seg_arr = annot_val.strip(' ').split('&')
                        if len(annot_val.strip(' '))> 0:
                            for result_annot_seg_item in result_annot_seg_arr:
                                result_annot_seg_arr = result_annot_seg_item.strip(' ').split('=')
                                result_annot_seg_slot = result_annot_seg_arr[0]
                                result_annot_seg_slot_val = result_annot_seg_arr[1]
                                
                                if result_annot_seg_slot_val == 'UNK': act_slot_value_response['request_slots'][result_annot_seg_slot] = 'UNK'
                                else: act_slot_value_response['inform_slots'][result_annot_seg_slot] = result_annot_seg_slot_val
                        else: # result={}
                            pass
                    else: # multi-choice or mc_list
                        annot_val_arr = annot_val.split('#')
                        act_slot_value_response['inform_slots'][annot_slot] = []
                        for annot_val_ele in annot_val_arr:
                            act_slot_value_response['inform_slots'][annot_slot].append(annot_val_ele)
                else: # single choice
                    if annot_slot in self.slot_set.keys():
                        if annot_val == 'UNK':
                            act_slot_value_response['request_slots'][annot_slot] = 'UNK'
                        else:
                            act_slot_value_response['inform_slots'][annot_slot] = annot_val
        
        return act_slot_value_response
    
    def generate_diaact_from_nl(self, string):
        """ Generate Dia_Act Form with NLU """
        
        agent_action = {}
        agent_action['diaact'] = 'UNK'
        agent_action['inform_slots'] = {}
        agent_action['request_slots'] = {}
        
        if len(string) > 0:
            agent_action = self.nlu_model.generate_dia_act(string)
        
        agent_action['nl'] = string 
        return agent_action
    
    def add_nl_to_action(self, agent_action):
        """ Add NL to Agent Dia_Act """
        
        if self.agent_input_mode == 1:
            if agent_action['act_slot_response']:
                agent_action['act_slot_response']['nl'] = ""
                user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_response'], 'agt')
                agent_action['act_slot_response']['nl'] = user_nlg_sentence
            elif agent_action['act_slot_value_response']:
                agent_action['act_slot_value_response']['nl'] = ""
                user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_value_response'], 'agt')
                agent_action['act_slot_response']['nl'] = user_nlg_sentence
                