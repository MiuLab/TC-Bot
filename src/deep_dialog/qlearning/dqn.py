'''
Created on Jun 18, 2016

@author: xiul
'''

from .utils import *


class DQN:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.model = {}
        # input-hidden
        self.model['Wxh'] = initWeight(input_size, hidden_size)
        self.model['bxh'] = np.zeros((1, hidden_size))
      
        # hidden-output
        self.model['Wd'] = initWeight(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['Wxh', 'bxh', 'Wd', 'bd']
        self.regularize = ['Wxh', 'Wd']

        self.step_cache = {}
        

    def getStruct(self):
        return {'model': self.model, 'update': self.update, 'regularize': self.regularize}
    
    
    """Activation Function: Sigmoid, or tanh, or ReLu"""
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        active_func = params.get('activation_func', 'relu')
 
        # input layer to hidden layer
        Wxh = self.model['Wxh']
        bxh = self.model['bxh']
        Xsh = Xs.dot(Wxh) + bxh
        
        hidden_size = self.model['Wd'].shape[0] # size of hidden layer
        H = np.zeros((1, hidden_size)) # hidden layer representation
        
        if active_func == 'sigmoid':
            H = 1/(1+np.exp(-Xsh))
        elif active_func == 'tanh':
            H = np.tanh(Xsh)
        elif active_func == 'relu': # ReLU 
            H = np.maximum(Xsh, 0)
        else: # no activation function
            H = Xsh
                
        # decoder at the end; hidden layer to output layer
        Wd = self.model['Wd']
        bd = self.model['bd']
        Y = H.dot(Wd) + bd
        
        # cache the values in forward pass, we expect to do a backward pass
        cache = {}
        if not predict_mode:
            cache['Wxh'] = Wxh
            cache['Wd'] = Wd
            cache['Xs'] = Xs
            cache['Xsh'] = Xsh
            cache['H'] = H
            
            cache['bxh'] = bxh
            cache['bd'] = bd
            cache['activation_func'] = active_func
            
            cache['Y'] = Y
            
        return Y, cache
    
    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        H = cache['H']
        Xs = cache['Xs']
        Xsh = cache['Xsh']
        Wxh = cache['Wxh']
 
        active_func = cache['activation_func']
        n,d = H.shape
        
        dH = dY.dot(Wd.transpose())
        # backprop the decoder
        dWd = H.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims=True)
        
        dXsh = np.zeros(Xsh.shape)
        dXs = np.zeros(Xs.shape)
        
        if active_func == 'sigmoid':
            dH = (H-H**2)*dH
        elif active_func == 'tanh':
            dH = (1-H**2)*dH
        elif active_func == 'relu':
            dH = (H>0)*dH # backprop ReLU
        else:
            dH = dH
              
        # backprop to the input-hidden connection
        dWxh = Xs.transpose().dot(dH)
        dbxh = np.sum(dH, axis=0, keepdims = True)
        
        # backprop to the input
        dXsh = dH
        dXs = dXsh.dot(Wxh.transpose())
        
        return {'Wd': dWd, 'bd': dbd, 'Wxh':dWxh, 'bxh':dbxh}
    
    
    """batch Forward & Backward Pass"""
    def batchForward(self, batch, params, predict_mode = False):
        caches = []
        Ys = []
        for i,x in enumerate(batch):
            Xs = np.array([x['cur_states']], dtype=float)
            
            Y, out_cache = self.fwdPass(Xs, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)
           
        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache
    
    def batchDoubleForward(self, batch, params, clone_dqn, predict_mode = False):
        caches = []
        Ys = []
        tYs = []
        
        for i,x in enumerate(batch):
            Xs = x[0]
            Y, out_cache = self.fwdPass(Xs, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)
            
            tXs = x[3]
            tY, t_cache = clone_dqn.fwdPass(tXs, params, predict_mode = False)
                
            tYs.append(tY)
            
        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache, tYs
    
    def batchBackward(self, dY, cache):
        caches = cache['caches']
        
        grads = {}
        for i in xrange(len(caches)):
            single_cache = caches[i]
            local_grads = self.bwdPass(dY[i], single_cache)
            mergeDicts(grads, local_grads) # add up the gradients wrt model parameters
            
        return grads


    """ cost function, returns cost and gradients for model """
    def costFunc(self, batch, params, clone_dqn):
        regc = params.get('reg_cost', 1e-3)
        gamma = params.get('gamma', 0.9)
        
        # batch forward
        Ys, caches, tYs = self.batchDoubleForward(batch, params, clone_dqn, predict_mode = False)
        
        loss_cost = 0.0
        dYs = []
        for i,x in enumerate(batch):
            Y = Ys[i]
            nY = tYs[i]
            
            action = np.array(x[1], dtype=int)
            reward = np.array(x[2], dtype=float)
            
            n_action = np.nanargmax(nY[0])
            max_next_y = nY[0][n_action]
            
            eposide_terminate = x[4]
            
            target_y = reward
            if eposide_terminate != True: target_y += gamma*max_next_y
            
            pred_y = Y[0][action]
            
            nY = np.zeros(nY.shape)
            nY[0][action] = target_y
            Y = np.zeros(Y.shape)
            Y[0][action] = pred_y
            
            # Cost Function
            loss_cost += (target_y - pred_y)**2
                
            dY = -(nY - Y)
            #dY = np.minimum(dY, 1)
            #dY = np.maximum(dY, -1)
            dYs.append(dY)
        
        # backprop the RNN
        grads = self.batchBackward(dYs, caches)
        
        # add L2 regularization cost and gradients
        reg_cost = 0.0
        if regc > 0:    
            for p in self.regularize:
                mat = self.model[p]
                reg_cost += 0.5*regc*np.sum(mat*mat)
                grads[p] += regc*mat

        # normalize the cost and gradient by the batch size
        batch_size = len(batch)
        reg_cost /= batch_size
        loss_cost /= batch_size
        for k in grads: grads[k] /= batch_size

        out = {}
        out['cost'] = {'reg_cost' : reg_cost, 'loss_cost' : loss_cost, 'total_cost' : loss_cost + reg_cost}
        out['grads'] = grads
        return out


    """ A single batch """
    def singleBatch(self, batch, params, clone_dqn):
        learning_rate = params.get('learning_rate', 0.001)
        decay_rate = params.get('decay_rate', 0.999)
        momentum = params.get('momentum', 0.1)
        grad_clip = params.get('grad_clip', -1e-3)
        smooth_eps = params.get('smooth_eps', 1e-8)
        sdg_type = params.get('sdgtype', 'rmsprop')
        activation_func = params.get('activation_func', 'relu')
        
        for u in self.update:
            if not u in self.step_cache: 
                self.step_cache[u] = np.zeros(self.model[u].shape)
        
        cg = self.costFunc(batch, params, clone_dqn)
        
        cost = cg['cost']
        grads = cg['grads']
        
        # clip gradients if needed
        if activation_func.lower() == 'relu':
            if grad_clip > 0:
                for p in self.update:
                    if p in grads:
                        grads[p] = np.minimum(grads[p], grad_clip)
                        grads[p] = np.maximum(grads[p], -grad_clip)
        
        # perform parameter update
        for p in self.update:
            if p in grads:
                if sdg_type == 'vanilla':
                    if momentum > 0:
                        dx = momentum*self.step_cache[p] - learning_rate*grads[p]
                    else:
                        dx = -learning_rate*grads[p]
                    self.step_cache[p] = dx
                elif sdg_type == 'rmsprop':
                    self.step_cache[p] = self.step_cache[p]*decay_rate + (1.0-decay_rate)*grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                elif sdg_type == 'adgrad':
                    self.step_cache[p] += grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                    
                self.model[p] += dx

        out = {}
        out['cost'] = cost
        return out
    
    """ prediction """
    def predict(self, Xs, params, **kwargs):
        Ys, caches = self.fwdPass(Xs, params, predict_model=True)
        pred_action = np.argmax(Ys)
        
        return pred_action
