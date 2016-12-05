'''
Created on Jun 13, 2016

An Bidirectional LSTM Seq2Seq model

@author: xiul
'''

from .seq_seq import SeqToSeq
from .utils import *


class biLSTM(SeqToSeq):
    def __init__(self, input_size, hidden_size, output_size):
        self.model = {}
        # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        self.model['WLSTM'] = initWeights(input_size + hidden_size + 1, 4*hidden_size)
        self.model['bWLSTM'] = initWeights(input_size + hidden_size + 1, 4*hidden_size)
        
        # Hidden-Output Connections
        self.model['Wd'] = initWeights(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))
        
        # Backward Hidden-Output Connections
        self.model['bWd'] = initWeights(hidden_size, output_size)*0.1
        self.model['bbd'] = np.zeros((1, output_size))

        self.update = ['WLSTM', 'bWLSTM', 'Wd', 'bd', 'bWd', 'bbd']
        self.regularize = ['WLSTM', 'bWLSTM', 'Wd', 'bWd']

        self.step_cache = {}
        
    """ Activation Function: Sigmoid, or tanh, or ReLu """
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        
        Ws = Xs['word_vectors']
        
        WLSTM = self.model['WLSTM']
        bWLSTM = self.model['bWLSTM']
        
        n, xd = Ws.shape
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((n, d))
        IFOG = np.zeros((n, 4*d))
        IFOGf = np.zeros((n, 4*d)) # after nonlinearity
        Cellin = np.zeros((n, d))
        Cellout = np.zeros((n, d))
        
        # backward
        bHin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
        bHout = np.zeros((n, d))
        bIFOG = np.zeros((n, 4*d))
        bIFOGf = np.zeros((n, 4*d)) # after nonlinearity
        bCellin = np.zeros((n, d))
        bCellout = np.zeros((n, d))
        
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

            # backward hidden layer
            b_t = n-1-t
            bprev = np.zeros(d) if t == 0 else bHout[b_t+1]
            bHin[b_t, 0] = 1
            bHin[b_t, 1:1+xd] = Ws[b_t]
            bHin[b_t, 1+xd:] = bprev
            
            bIFOG[b_t] = bHin[b_t].dot(bWLSTM)
            bIFOGf[b_t, :3*d] = 1/(1+np.exp(-bIFOG[b_t, :3*d]))
            bIFOGf[b_t, 3*d:] = np.tanh(bIFOG[b_t, 3*d:])
            
            bCellin[b_t] = bIFOGf[b_t, :d] * bIFOGf[b_t, 3*d:]
            if t>0: bCellin[b_t] += bIFOGf[b_t, d:2*d] * bCellin[b_t+1]
            
            bCellout[b_t] = np.tanh(bCellin[b_t])
            bHout[b_t] = bIFOGf[b_t, 2*d:3*d]*bCellout[b_t]
            
        Wd = self.model['Wd']
        bd = self.model['bd']
        fY = Hout.dot(Wd)+bd
        
        bWd = self.model['bWd']
        bbd = self.model['bbd']
        bY = bHout.dot(bWd)+bbd
        
        Y = fY + bY
            
        cache = {}
        if not predict_mode:
            cache['WLSTM'] = WLSTM
            cache['Hout'] = Hout
            cache['Wd'] = Wd
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['Cellin'] = Cellin
            cache['Cellout'] = Cellout
            cache['Hin'] = Hin
            
            cache['bWLSTM'] = bWLSTM
            cache['bHout'] = bHout
            cache['bWd'] = bWd
            cache['bIFOGf'] = bIFOGf
            cache['bIFOG'] = bIFOG
            cache['bCellin'] = bCellin
            cache['bCellout'] = bCellout
            cache['bHin'] = bHin
            
            cache['Ws'] = Ws
            
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
        
        bWd = cache['bWd']
        bHout = cache['bHout']
        bIFOG = cache['bIFOG']
        bIFOGf = cache['bIFOGf']
        bCellin = cache['bCellin']
        bCellout = cache['bCellout']
        bHin = cache['bHin']
        bWLSTM = cache['bWLSTM']
        
        n,d = Hout.shape

        # backprop the hidden-output layer
        dWd = Hout.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims = True)
        dHout = dY.dot(Wd.transpose())
        
        # backprop the backward hidden-output layer
        dbWd = bHout.transpose().dot(dY)
        dbbd = np.sum(dY, axis=0, keepdims = True)
        dbHout = dY.dot(bWd.transpose())
        
        # backprop the LSTM (forward layer)
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dCellin = np.zeros(Cellin.shape)
        dCellout = np.zeros(Cellout.shape)
        
        # backward-layer
        dbIFOG = np.zeros(bIFOG.shape)
        dbIFOGf = np.zeros(bIFOGf.shape)
        dbWLSTM = np.zeros(bWLSTM.shape)
        dbHin = np.zeros(bHin.shape)
        dbCellin = np.zeros(bCellin.shape)
        dbCellout = np.zeros(bCellout.shape)
        
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
      
            if t>0: dHout[t-1] += dHin[t, 1+Ws.shape[1]:]
            
            # Backward Layer
            b_t = n-1-t
            dbIFOGf[b_t, 2*d:3*d] = bCellout[b_t] * dbHout[b_t] # output gate
            dbCellout[b_t] = bIFOGf[b_t, 2*d:3*d] * dbHout[b_t] # dCellout
            
            dbCellin[b_t] += (1-bCellout[b_t]**2) * dbCellout[b_t]
            
            if t>0: # dcell
                dbIFOGf[b_t, d:2*d] = bCellin[b_t+1] * dbCellin[b_t] # forgot gate
                dbCellin[b_t+1] += bIFOGf[b_t, d:2*d] * dbCellin[b_t]
            
            dbIFOGf[b_t, :d] = bIFOGf[b_t, 3*d:] * dbCellin[b_t] # input gate
            dbIFOGf[b_t, 3*d:] = bIFOGf[b_t, :d] * dbCellin[b_t]
            
            # backprop activation functions
            dbIFOG[b_t, 3*d:] = (1-bIFOGf[b_t, 3*d:]**2) * dbIFOGf[b_t, 3*d:]
            by = bIFOGf[b_t, :3*d]
            dbIFOG[b_t, :3*d] = (by*(1-by)) * dbIFOGf[b_t, :3*d]
            
            dbWLSTM += np.outer(bHin[b_t], dbIFOG[b_t])
            dbHin[b_t] = dbIFOG[b_t].dot(bWLSTM.transpose())
      
            if t>0: dbHout[b_t+1] += dbHin[b_t, 1+Ws.shape[1]:]
                
        return {'WLSTM':dWLSTM, 'Wd':dWd, 'bd':dbd, 'bWLSTM':dbWLSTM, 'bWd':dbWd, 'bbd':dbbd}