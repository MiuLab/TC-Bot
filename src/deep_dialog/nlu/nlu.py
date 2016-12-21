'''
Created on Jul 13, 2016

@author: xiul
'''

import cPickle as pickle
import copy
import numpy as np

from lstm import lstm
from bi_lstm import biLSTM


class nlu:
    def __init__(self):
        pass
    
    def generate_dia_act(self, annot):
        """ generate the Dia-Act with NLU model """
        
        if len(annot) > 0:
            tmp_annot = annot.strip('.').strip('?').strip(',').strip('!') 
            
            rep = self.parse_str_to_vector(tmp_annot)
            Ys, cache = self.model.fwdPass(rep, self.params, predict_model=True) # default: True
            
            maxes = np.amax(Ys, axis=1, keepdims=True)
            e = np.exp(Ys - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)
            
            # special handling with intent label
            for tag_id in self.inverse_tag_dict.keys():
                if self.inverse_tag_dict[tag_id].startswith('B-') or self.inverse_tag_dict[tag_id].startswith('I-') or self.inverse_tag_dict[tag_id] == 'O':
                    probs[-1][tag_id] = 0
            
            pred_words_indices = np.nanargmax(probs, axis=1)
            pred_tags = [self.inverse_tag_dict[index] for index in pred_words_indices]
            
            diaact = self.parse_nlu_to_diaact(pred_tags, tmp_annot)
            return diaact
        else:
            return None

    
    def load_nlu_model(self, model_path):
        """ load the trained NLU model """  
        
        model_params = pickle.load(open(model_path, 'rb'))
    
        hidden_size = model_params['model']['Wd'].shape[0]
        output_size = model_params['model']['Wd'].shape[1]
    
        if model_params['params']['model'] == 'lstm': # lstm_
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = lstm(input_size, hidden_size, output_size)
        elif model_params['params']['model'] == 'bi_lstm': # bi_lstm
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = biLSTM(input_size, hidden_size, output_size)
           
        rnnmodel.model = copy.deepcopy(model_params['model'])
        
        self.model = rnnmodel
        self.word_dict = copy.deepcopy(model_params['word_dict'])
        self.slot_dict = copy.deepcopy(model_params['slot_dict'])
        self.act_dict = copy.deepcopy(model_params['act_dict'])
        self.tag_set = copy.deepcopy(model_params['tag_set'])
        self.params = copy.deepcopy(model_params['params'])
        self.inverse_tag_dict = {self.tag_set[k]:k for k in self.tag_set.keys()}
        
           
    def parse_str_to_vector(self, string):
        """ Parse string into vector representations """
        
        tmp = 'BOS ' + string + ' EOS'
        words = tmp.lower().split(' ')
        
        vecs = np.zeros((len(words), len(self.word_dict)))
        for w_index, w in enumerate(words):
            if w.endswith(',') or w.endswith('?'): w = w[0:-1]
            if w in self.word_dict.keys():
                vecs[w_index][self.word_dict[w]] = 1
            else: vecs[w_index][self.word_dict['unk']] = 1
        
        rep = {}
        rep['word_vectors'] = vecs
        rep['raw_seq'] = string
        return rep

    def parse_nlu_to_diaact(self, nlu_vector, string):
        """ Parse BIO and Intent into Dia-Act """
        
        tmp = 'BOS ' + string + ' EOS'
        words = tmp.lower().split(' ')
    
        diaact = {}
        diaact['diaact'] = "inform"
        diaact['request_slots'] = {}
        diaact['inform_slots'] = {}
        
        intent = nlu_vector[-1]
        index = 1
        pre_tag = nlu_vector[0]
        pre_tag_index = 0
    
        slot_val_dict = {}
    
        while index<(len(nlu_vector)-1): # except last Intent tag
            cur_tag = nlu_vector[index]
            if cur_tag == 'O' and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('I-'):
                if cur_tag.split('-')[1] != pre_tag.split('-')[1]:           
                    slot = pre_tag.split('-')[1]
                    slot_val_str = ' '.join(words[pre_tag_index:index])
                    slot_val_dict[slot] = slot_val_str
            elif cur_tag == 'O' and pre_tag.startswith('I-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
               
            if cur_tag.startswith('B-'): pre_tag_index = index
        
            pre_tag = cur_tag
            index += 1
    
        if cur_tag.startswith('B-') or cur_tag.startswith('I-'):
            slot = cur_tag.split('-')[1]
            slot_val_str = ' '.join(words[pre_tag_index:-1])
            slot_val_dict[slot] = slot_val_str
    
        if intent != 'null':
            arr = intent.split('+')
            diaact['diaact'] = arr[0]
            diaact['request_slots'] = {}
            for ele in arr[1:]: 
                #request_slots.append(ele)
                diaact['request_slots'][ele] = 'UNK'
        
        diaact['inform_slots'] = slot_val_dict
         
        # add rule here
        for slot in diaact['inform_slots'].keys():
            slot_val = diaact['inform_slots'][slot]
            if slot_val.startswith('bos'): 
                slot_val = slot_val.replace('bos', '', 1)
                diaact['inform_slots'][slot] = slot_val.strip(' ')
        
        self.refine_diaact_by_rules(diaact)
        return diaact

    def refine_diaact_by_rules(self, diaact):
        """ refine the dia_act by rules """
        
        # rule for taskcomplete
        if 'request_slots' in diaact.keys():
            if 'taskcomplete' in diaact['request_slots'].keys():
                del diaact['request_slots']['taskcomplete']
                diaact['inform_slots']['taskcomplete'] = 'PLACEHOLDER'
        
            # rule for request
            if len(diaact['request_slots'])>0: diaact['diaact'] = 'request'
    
    
    
    
    def diaact_penny_string(self, dia_act):
        """ Convert the Dia-Act into penny string """
        
        penny_str = ""
        penny_str = dia_act['diaact'] + "("
        for slot in dia_act['request_slots'].keys():
            penny_str += slot + ";"
    
        for slot in dia_act['inform_slots'].keys():
            slot_val_str = slot + "="
            if len(dia_act['inform_slots'][slot]) == 1:
                slot_val_str += dia_act['inform_slots'][slot][0]
            else:
                slot_val_str += "{"
                for slot_val in dia_act['inform_slots'][slot]:
                    slot_val_str += slot_val + "#"
                slot_val_str = slot_val_str[:-1]
                slot_val_str += "}"
            penny_str += slot_val_str + ";"
    
        if penny_str[-1] == ";": penny_str = penny_str[:-1]
        penny_str += ")"
        return penny_str