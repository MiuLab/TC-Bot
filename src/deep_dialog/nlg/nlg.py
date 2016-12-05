'''
Created on Oct 17, 2016

--dia_act_nl_pairs.v6.json: agt and usr have their own NL.


@author: xiul
'''

import cPickle as pickle
import copy, argparse, json
import numpy as np

from deep_dialog import dialog_config
from deep_dialog.nlg.lstm_decoder_tanh import lstm_decoder_tanh


class nlg:
    def __init__(self):
        pass
    
    def post_process(self, pred_template, slot_val_dict, slot_dict):
        """ post_process to fill the slot in the template sentence """
        
        sentence = pred_template
        suffix = "_PLACEHOLDER"
        
        for slot in slot_val_dict.keys():
            slot_vals = slot_val_dict[slot]
            slot_placeholder = slot + suffix
            if slot == 'result' or slot == 'numberofpeople': continue
            if slot_vals == dialog_config.NO_VALUE_MATCH: continue
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence
                
        if 'numberofpeople' in slot_val_dict.keys():
            slot_vals = slot_val_dict['numberofpeople']
            slot_placeholder = 'numberofpeople' + suffix
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence
    
        for slot in slot_dict.keys():
            slot_placeholder = slot + suffix
            tmp_sentence = sentence.replace(slot_placeholder, '')
            sentence = tmp_sentence
            
        return sentence

    
    def convert_diaact_to_nl(self, dia_act, turn_msg):
        """ Convert Dia_Act into NL: Rule + Model """
        
        sentence = ""
        boolean_in = False
        
        # remove I do not care slot in task(complete)
        if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] != dialog_config.NO_VALUE_MATCH:
            inform_slot_set = dia_act['inform_slots'].keys()
            for slot in inform_slot_set:
                if dia_act['inform_slots'][slot] == dialog_config.I_DO_NOT_CARE: del dia_act['inform_slots'][slot]
        
        if dia_act['diaact'] in self.diaact_nl_pairs['dia_acts'].keys():
            for ele in self.diaact_nl_pairs['dia_acts'][dia_act['diaact']]:
                if set(ele['inform_slots']) == set(dia_act['inform_slots'].keys()) and set(ele['request_slots']) == set(dia_act['request_slots'].keys()):
                    sentence = self.diaact_to_nl_slot_filling(dia_act, ele['nl'][turn_msg])
                    boolean_in = True
                    break
        
        if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] == dialog_config.NO_VALUE_MATCH:
            sentence = "Oh sorry, there is no ticket available."
        
        if boolean_in == False: sentence = self.translate_diaact(dia_act)
        return sentence
        
        
    def translate_diaact(self, dia_act):
        """ prepare the diaact into vector representation, and generate the sentence by Model """
        
        word_dict = self.word_dict
        template_word_dict = self.template_word_dict
        act_dict = self.act_dict
        slot_dict = self.slot_dict
        inverse_word_dict = self.inverse_word_dict
    
        act_rep = np.zeros((1, len(act_dict)))
        act_rep[0, act_dict[dia_act['diaact']]] = 1.0
    
        slot_rep_bit = 2
        slot_rep = np.zeros((1, len(slot_dict)*slot_rep_bit)) 
    
        suffix = "_PLACEHOLDER"
        if self.params['dia_slot_val'] == 2 or self.params['dia_slot_val'] == 3:
            word_rep = np.zeros((1, len(template_word_dict)))
            words = np.zeros((1, len(template_word_dict)))
            words[0, template_word_dict['s_o_s']] = 1.0
        else:
            word_rep = np.zeros((1, len(word_dict)))
            words = np.zeros((1, len(word_dict)))
            words[0, word_dict['s_o_s']] = 1.0
    
        for slot in dia_act['inform_slots'].keys():
            slot_index = slot_dict[slot]
            slot_rep[0, slot_index*slot_rep_bit] = 1.0
        
            for slot_val in dia_act['inform_slots'][slot]:
                if self.params['dia_slot_val'] == 2:
                    slot_placeholder = slot + suffix
                    if slot_placeholder in template_word_dict.keys():
                        word_rep[0, template_word_dict[slot_placeholder]] = 1.0
                elif self.params['dia_slot_val'] == 1:
                    if slot_val in word_dict.keys():
                        word_rep[0, word_dict[slot_val]] = 1.0
                    
        for slot in dia_act['request_slots'].keys():
            slot_index = slot_dict[slot]
            slot_rep[0, slot_index*slot_rep_bit + 1] = 1.0
    
        if self.params['dia_slot_val'] == 0 or self.params['dia_slot_val'] == 3:
            final_representation = np.hstack([act_rep, slot_rep])
        else: # dia_slot_val = 1, 2
            final_representation = np.hstack([act_rep, slot_rep, word_rep])
    
        dia_act_rep = {}
        dia_act_rep['diaact'] = final_representation
        dia_act_rep['words'] = words
    
        #pred_ys, pred_words = nlg_model['model'].forward(inverse_word_dict, dia_act_rep, nlg_model['params'], predict_model=True)
        pred_ys, pred_words = self.model.beam_forward(inverse_word_dict, dia_act_rep, self.params, predict_model=True)
        pred_sentence = ' '.join(pred_words[:-1])
        sentence = self.post_process(pred_sentence, dia_act['inform_slots'], slot_dict)
            
        return sentence
    
    
    def load_nlg_model(self, model_path):
        """ load the trained NLG model """  
        
        model_params = pickle.load(open(model_path, 'rb'))
    
        hidden_size = model_params['model']['Wd'].shape[0]
        output_size = model_params['model']['Wd'].shape[1]
    
        if model_params['params']['model'] == 'lstm_tanh': # lstm_tanh
            diaact_input_size = model_params['model']['Wah'].shape[0]
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = lstm_decoder_tanh(diaact_input_size, input_size, hidden_size, output_size)
        
        rnnmodel.model = copy.deepcopy(model_params['model'])
        model_params['params']['beam_size'] = dialog_config.nlg_beam_size
        
        self.model = rnnmodel
        self.word_dict = copy.deepcopy(model_params['word_dict'])
        self.template_word_dict = copy.deepcopy(model_params['template_word_dict'])
        self.slot_dict = copy.deepcopy(model_params['slot_dict'])
        self.act_dict = copy.deepcopy(model_params['act_dict'])
        self.inverse_word_dict = {self.template_word_dict[k]:k for k in self.template_word_dict.keys()}
        self.params = copy.deepcopy(model_params['params'])
        
    
    def diaact_to_nl_slot_filling(self, dia_act, template_sentence):
        """ Replace the slots with its values """
        
        sentence = template_sentence
        counter = 0
        for slot in dia_act['inform_slots'].keys():
            slot_val = dia_act['inform_slots'][slot]
            if slot_val == dialog_config.NO_VALUE_MATCH:
                sentence = slot + " is not available!"
                break
            elif slot_val == dialog_config.I_DO_NOT_CARE:
                counter += 1
                sentence = sentence.replace('$'+slot+'$', '', 1)
                continue
            
            sentence = sentence.replace('$'+slot+'$', slot_val, 1)
        
        if counter > 0 and counter == len(dia_act['inform_slots']):
            sentence = dialog_config.I_DO_NOT_CARE
        
        return sentence
    
    
    def load_predefine_act_nl_pairs(self, path):
        """ Load some pre-defined Dia_Act&NL Pairs from file """
        
        self.diaact_nl_pairs = json.load(open(path, 'rb'))
        
        for key in self.diaact_nl_pairs['dia_acts'].keys():
            for ele in self.diaact_nl_pairs['dia_acts'][key]:
                ele['nl']['usr'] = ele['nl']['usr'].encode('utf-8') # encode issue
                ele['nl']['agt'] = ele['nl']['agt'].encode('utf-8') # encode issue


def main(params):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print ("User Simulator Parameters:")
    print (json.dumps(params, indent=2))

    main(params)
