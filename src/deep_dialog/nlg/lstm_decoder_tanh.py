'''
Created on Jun 13, 2016

An LSTM decoder - add tanh after cell before output gate

@author: xiul
'''

from .decoder import decoder
from .utils import *


class lstm_decoder_tanh(decoder):
    def __init__(self, diaact_input_size, input_size, hidden_size, output_size):
        self.model = {}
        # connections from diaact to hidden layer
        self.model['Wah'] = initWeights(diaact_input_size, 4*hidden_size)
        self.model['bah'] = np.zeros((1, 4*hidden_size))
        
        # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        self.model['WLSTM'] = initWeights(input_size + hidden_size + 1, 4*hidden_size)
        # Hidden-Output Connections
        self.model['Wd'] = initWeights(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['Wah', 'bah', 'WLSTM', 'Wd', 'bd']
        self.regularize = ['Wah', 'WLSTM', 'Wd']

        self.step_cache = {}
        
    """ Activation Function: Sigmoid, or tanh, or ReLu """
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        feed_recurrence = params.get('feed_recurrence', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        n, xd = Ws.shape
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((n, d))
        IFOG = np.zeros((n, 4*d))
        IFOGf = np.zeros((n, 4*d)) # after nonlinearity
        Cellin = np.zeros((n, d))
        Cellout = np.zeros((n, d))
    
        for t in xrange(n):
            prev = np.zeros(d) if t==0 else Hout[t-1]
            Hin[t,0] = 1 # bias
            Hin[t, 1:1+xd] = Ws[t]
            Hin[t, 1+xd:] = prev
            
            # compute all gate activations. dots:
            IFOG[t] = Hin[t].dot(WLSTM)
            
            # add diaact vector here
            if feed_recurrence == 0:
                if t == 0: IFOG[t] += Dsh[0]
            else:
                IFOG[t] += Dsh[0]

            IFOGf[t, :3*d] = 1/(1+np.exp(-IFOG[t, :3*d])) # sigmoids; these are three gates
            IFOGf[t, 3*d:] = np.tanh(IFOG[t, 3*d:]) # tanh for input value
            
            Cellin[t] = IFOGf[t, :d] * IFOGf[t, 3*d:]
            if t>0: Cellin[t] += IFOGf[t, d:2*d]*Cellin[t-1]
            
            Cellout[t] = np.tanh(Cellin[t])
            
            Hout[t] = IFOGf[t, 2*d:3*d] * Cellout[t]

        Wd = self.model['Wd']
        bd = self.model['bd']
            
        Y = Hout.dot(Wd)+bd
            
        cache = {}
        if not predict_mode:
            cache['WLSTM'] = WLSTM
            cache['Hout'] = Hout
            cache['WLSTM'] = WLSTM
            cache['Wd'] = Wd
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['Cellin'] = Cellin
            cache['Cellout'] = Cellout
            cache['Ws'] = Ws
            cache['Ds'] = Ds
            cache['Hin'] = Hin
            cache['Dsh'] = Dsh
            cache['Wah'] = Wah
            cache['feed_recurrence'] = feed_recurrence
            
        return Y, cache
    
    """ Forward pass on prediction """
    def forward(self, dict, Xs, params, **kwargs):
        max_len = params.get('max_len', 30)
        feed_recurrence = params.get('feed_recurrence', 0)
        decoder_sampling = params.get('decoder_sampling', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        xd = Ws.shape[1]
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((1, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((1, d))
        IFOG = np.zeros((1, 4*d))
        IFOGf = np.zeros((1, 4*d)) # after nonlinearity
        Cellin = np.zeros((1, d))
        Cellout = np.zeros((1, d))
        
        Wd = self.model['Wd']
        bd = self.model['bd']
        
        Hin[0,0] = 1 # bias
        Hin[0,1:1+xd] = Ws[0]
        
        IFOG[0] = Hin[0].dot(WLSTM)
        IFOG[0] += Dsh[0]
        
        IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
        IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
        Cellin[0] = IFOGf[0, :d] * IFOGf[0, 3*d:]
        Cellout[0] = np.tanh(Cellin[0])
        Hout[0] = IFOGf[0, 2*d:3*d] * Cellout[0]
        
        pred_y = []
        pred_words = []
        
        Y = Hout.dot(Wd) + bd
        maxes = np.amax(Y, axis=1, keepdims=True)
        e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
        probs = e/np.sum(e, axis=1, keepdims=True)
            
        if decoder_sampling == 0: # sampling or argmax
            pred_y_index = np.nanargmax(Y)
        else:
            pred_y_index = np.random.choice(Y.shape[1], 1, p=probs[0])[0]
        pred_y.append(pred_y_index)
        pred_words.append(dict[pred_y_index])
        
        time_stamp = 0
        while True:
            if dict[pred_y_index] == 'e_o_s' or time_stamp >= max_len: break
            
            X = np.zeros(xd)
            X[pred_y_index] = 1
            Hin[0,0] = 1 # bias
            Hin[0,1:1+xd] = X
            Hin[0, 1+xd:] = Hout[0]
            
            IFOG[0] = Hin[0].dot(WLSTM)
            if feed_recurrence == 1:
                IFOG[0] += Dsh[0]
        
            IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
            IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
            C = IFOGf[0, :d]*IFOGf[0, 3*d:]
            Cellin[0] = C + IFOGf[0, d:2*d]*Cellin[0]
            Cellout[0] = np.tanh(Cellin[0])
            Hout[0] = IFOGf[0, 2*d:3*d]*Cellout[0]
            
            Y = Hout.dot(Wd) + bd
            maxes = np.amax(Y, axis=1, keepdims=True)
            e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            
            if decoder_sampling == 0:
                pred_y_index = np.nanargmax(Y)
            else:
                pred_y_index = np.random.choice(Y.shape[1], 1, p=probs[0])[0]
            pred_y.append(pred_y_index)
            pred_words.append(dict[pred_y_index])
            
            time_stamp += 1
            
        return pred_y, pred_words
    
    """ Forward pass on prediction with Beam Search """
    def beam_forward(self, dict, Xs, params, **kwargs):
        max_len = params.get('max_len', 30)
        feed_recurrence = params.get('feed_recurrence', 0)
        beam_size = params.get('beam_size', 10)
        decoder_sampling = params.get('decoder_sampling', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        xd = Ws.shape[1]
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((1, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((1, d))
        IFOG = np.zeros((1, 4*d))
        IFOGf = np.zeros((1, 4*d)) # after nonlinearity
        Cellin = np.zeros((1, d))
        Cellout = np.zeros((1, d))
        
        Wd = self.model['Wd']
        bd = self.model['bd']
        
        Hin[0,0] = 1 # bias
        Hin[0,1:1+xd] = Ws[0]
        
        IFOG[0] = Hin[0].dot(WLSTM)
        IFOG[0] += Dsh[0]
        
        IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
        IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
        Cellin[0] = IFOGf[0, :d] * IFOGf[0, 3*d:]
        Cellout[0] = np.tanh(Cellin[0])
        Hout[0] = IFOGf[0, 2*d:3*d] * Cellout[0]
        
        # keep a beam here
        beams = [] 
        
        Y = Hout.dot(Wd) + bd
        maxes = np.amax(Y, axis=1, keepdims=True)
        e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
        probs = e/np.sum(e, axis=1, keepdims=True)
        
        # add beam search here
        if decoder_sampling == 0: # no sampling
            beam_candidate_t = (-probs[0]).argsort()[:beam_size]
        else:
            beam_candidate_t = np.random.choice(Y.shape[1], beam_size, p=probs[0])
        #beam_candidate_t = (-probs[0]).argsort()[:beam_size]
        for ele in beam_candidate_t:
            beams.append((np.log(probs[0][ele]), [ele], [dict[ele]], Hout[0], Cellin[0]))
        
        #beams.sort(key=lambda x:x[0], reverse=True)
        #beams.sort(reverse = True)
        
        time_stamp = 0
        while True:
            beam_candidates = []
            for b in beams:
                log_prob = b[0]
                pred_y_index = b[1][-1]
                cell_in = b[4]
                hout_prev = b[3]
                
                if b[2][-1] == "e_o_s": # this beam predicted end token. Keep in the candidates but don't expand it out any more
                    beam_candidates.append(b)
                    continue
        
                X = np.zeros(xd)
                X[pred_y_index] = 1
                Hin[0,0] = 1 # bias
                Hin[0,1:1+xd] = X
                Hin[0, 1+xd:] = hout_prev
                
                IFOG[0] = Hin[0].dot(WLSTM)
                if feed_recurrence == 1: IFOG[0] += Dsh[0]
        
                IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
                IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
                C = IFOGf[0, :d]*IFOGf[0, 3*d:]
                cell_in = C + IFOGf[0, d:2*d]*cell_in
                cell_out = np.tanh(cell_in)
                hout_prev = IFOGf[0, 2*d:3*d]*cell_out
                
                Y = hout_prev.dot(Wd) + bd
                maxes = np.amax(Y, axis=1, keepdims=True)
                e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
                probs = e/np.sum(e, axis=1, keepdims=True)
                
                if decoder_sampling == 0: # no sampling
                    beam_candidate_t = (-probs[0]).argsort()[:beam_size]
                else:
                    beam_candidate_t = np.random.choice(Y.shape[1], beam_size, p=probs[0])
                #beam_candidate_t = (-probs[0]).argsort()[:beam_size]
                for ele in beam_candidate_t:
                    beam_candidates.append((log_prob+np.log(probs[0][ele]), np.append(b[1], ele), np.append(b[2], dict[ele]), hout_prev, cell_in))
            
            beam_candidates.sort(key=lambda x:x[0], reverse=True)
            #beam_candidates.sort(reverse = True) # decreasing order
            beams = beam_candidates[:beam_size]
            time_stamp += 1

            if time_stamp >= max_len: break
        
        return beams[0][1], beams[0][2]
    
    """ Backward Pass """
    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        Hout = cache['Hout']
        IFOG = cache['IFOG']
        IFOGf = cache['IFOGf']
        Cellin = cache['Cellin']
        Cellout = cache['Cellout']
        Hin = cache['Hin']
        WLSTM = cache['WLSTM']
        Ws = cache['Ws']
        Ds = cache['Ds']
        Dsh = cache['Dsh']
        Wah = cache['Wah']
        feed_recurrence = cache['feed_recurrence']
        
        n,d = Hout.shape

        # backprop the hidden-output layer
        dWd = Hout.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims = True)
        dHout = dY.dot(Wd.transpose())

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dCellin = np.zeros(Cellin.shape)
        dCellout = np.zeros(Cellout.shape)
        dWs = np.zeros(Ws.shape)
        
        dDsh = np.zeros(Dsh.shape)
        
        for t in reversed(xrange(n)):
            dIFOGf[t,2*d:3*d] = Cellout[t] * dHout[t]
            dCellout[t] = IFOGf[t,2*d:3*d] * dHout[t]
            
            dCellin[t] += (1-Cellout[t]**2) * dCellout[t]
            
            if t>0:
                dIFOGf[t, d:2*d] = Cellin[t-1] * dCellin[t]
                dCellin[t-1] += IFOGf[t,d:2*d] * dCellin[t]
            
            dIFOGf[t, :d] = IFOGf[t,3*d:] * dCellin[t]
            dIFOGf[t,3*d:] = IFOGf[t, :d] * dCellin[t]
            
            # backprop activation functions
            dIFOG[t, 3*d:] = (1-IFOGf[t, 3*d:]**2) * dIFOGf[t, 3*d:]
            y = IFOGf[t, :3*d]
            dIFOG[t, :3*d] = (y*(1-y)) * dIFOGf[t, :3*d]
            
            # backprop matrix multiply
            dWLSTM += np.outer(Hin[t], dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())
      
            if t > 0: dHout[t-1] += dHin[t,1+Ws.shape[1]:]
            
            if feed_recurrence == 0:
                if t == 0: dDsh[t] = dIFOG[t]
            else: 
                dDsh[0] += dIFOG[t]
        
        # backprop to the diaact-hidden connections
        dWah = Ds.transpose().dot(dDsh)
        dbah = np.sum(dDsh, axis=0, keepdims = True)
             
        return {'Wah':dWah, 'bah':dbah, 'WLSTM':dWLSTM, 'Wd':dWd, 'bd':dbd}
    
    
    """ Batch data representation """
    def prepare_input_rep(self, ds, batch, params):
        batch_reps = []
        for i,x in enumerate(batch):
            batch_rep = {}
            
            vec = np.zeros((1, self.model['Wah'].shape[0]))
            vec[0][x['diaact_rep']] = 1
            for v in x['slotrep']:
                vec[0][v] = 1
            
            word_arr = x['sentence'].split(' ')
            word_vecs = np.zeros((len(word_arr), self.model['Wxh'].shape[0]))
            labels = [0] * (len(word_arr)-1)
            for w_index, w in enumerate(word_arr[:-1]):
                if w in ds.data['word_dict'].keys():
                    w_dict_index = ds.data['word_dict'][w]
                    word_vecs[w_index][w_dict_index] = 1
                
                if word_arr[w_index+1] in ds.data['word_dict'].keys():
                    labels[w_index] = ds.data['word_dict'][word_arr[w_index+1]] 
            
            batch_rep['diaact'] = vec
            batch_rep['words'] = word_vecs
            batch_rep['labels'] = labels
            batch_reps.append(batch_rep)
        return batch_reps