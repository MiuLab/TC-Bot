'''
Created on Jun 13, 2016

An LSTM decoder - add tanh after cell before output gate

@author: xiul
'''

from seq_seq import SeqToSeq
from .utils import *


class lstm(SeqToSeq):
    def __init__(self, input_size, hidden_size, output_size):
        self.model = {}
        # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        self.model['WLSTM'] = initWeights(input_size + hidden_size + 1, 4*hidden_size)
        # Hidden-Output Connections
        self.model['Wd'] = initWeights(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['WLSTM', 'Wd', 'bd']
        self.regularize = ['WLSTM', 'Wd']

        self.step_cache = {}
        
    """ Activation Function: Sigmoid, or tanh, or ReLu """
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        
        Ws = Xs['word_vectors']
        
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
            cache['Wd'] = Wd
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['Cellin'] = Cellin
            cache['Cellout'] = Cellout
            cache['Ws'] = Ws
            cache['Hin'] = Hin
            
        return Y, cache
    
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
      
            if t > 0: dHout[t-1] += dHin[t, 1+Ws.shape[1]:]
        
        #dXs = dXsh.dot(Wxh.transpose())  
        return {'WLSTM':dWLSTM, 'Wd':dWd, 'bd':dbd}