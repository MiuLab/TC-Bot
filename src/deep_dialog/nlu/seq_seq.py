'''
Created on Jun 13, 2016

@author: xiul
'''

from .utils import *
import time, os


class SeqToSeq:
    def __init__(self, input_size, hidden_size, output_size):
        pass
    
    def get_struct(self):
        return {'model': self.model, 'update': self.update, 'regularize': self.regularize}
    
    
    """ Forward Function"""
    def fwdPass(self, Xs, params, **kwargs):
        pass
    
    def bwdPass(self, dY, cache):
        pass
    
    
    """ Batch Forward & Backward Pass"""
    def batchForward(self, ds, batch, params, predict_mode = False):
        caches = []
        Ys = []
        for i,x in enumerate(batch):
            Y, out_cache = self.fwdPass(x, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)
           
        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache
    
    def batchBackward(self, dY, cache):
        caches = cache['caches']
        grads = {}
        for i in xrange(len(caches)):
            single_cache = caches[i]
            local_grads = self.bwdPass(dY[i], single_cache)
            mergeDicts(grads, local_grads) # add up the gradients wrt model parameters
            
        return grads


    """ Cost function, returns cost and gradients for model """
    def costFunc(self, ds, batch, params):
        regc = params['reg_cost'] # regularization cost
        
        # batch forward RNN
        Ys, caches = self.batchForward(ds, batch, params, predict_mode = False)
        
        loss_cost = 0.0
        smooth_cost = 1e-15
        dYs = []
        
        for i,x in enumerate(batch):
            labels = np.array(x['tags_rep'], dtype=int)
            
            # fetch the predicted probabilities
            Y = Ys[i]
            maxes = np.amax(Y, axis=1, keepdims=True)
            e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
            P = e/np.sum(e, axis=1, keepdims=True)
            
            # Cross-Entropy Cross Function
            loss_cost += -np.sum(np.log(smooth_cost + P[range(len(labels)), labels]))
            
            for iy,y in enumerate(labels):
                P[iy,y] -= 1 # softmax derivatives
            dYs.append(P)
            
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
    def singleBatch(self, ds, batch, params):
        learning_rate = params.get('learning_rate', 0.0)
        decay_rate = params.get('decay_rate', 0.999)
        momentum = params.get('momentum', 0)
        grad_clip = params.get('grad_clip', 1)
        smooth_eps = params.get('smooth_eps', 1e-8)
        sdg_type = params.get('sdgtype', 'rmsprop')

        for u in self.update:
            if not u in self.step_cache: 
                self.step_cache[u] = np.zeros(self.model[u].shape)
        
        cg = self.costFunc(ds, batch, params)
        
        cost = cg['cost']
        grads = cg['grads']
        
        # clip gradients if needed
        if params['activation_func'] == 'relu':
            if grad_clip > 0:
                for p in self.update:
                    if p in grads:
                        grads[p] = np.minimum(grads[p], grad_clip)
                        grads[p] = np.maximum(grads[p], -grad_clip)
        
        # perform parameter update
        for p in self.update:
            if p in grads:
                if sdg_type == 'vanilla':
                    if momentum > 0: dx = momentum*self.step_cache[p] - learning_rate*grads[p]
                    else: dx = -learning_rate*grads[p]
                    self.step_cache[p] = dx
                elif sdg_type == 'rmsprop':
                    self.step_cache[p] = self.step_cache[p]*decay_rate + (1.0-decay_rate)*grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                elif sdg_type == 'adgrad':
                    self.step_cache[p] += grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                    
                self.model[p] += dx

        # create output dict and return
        out = {}
        out['cost'] = cost
        return out
    
    
    """ Evaluate on the dataset[split] """
    def eval(self, ds, split, params):
        acc = 0
        total = 0
        
        total_cost = 0.0
        smooth_cost = 1e-15
        
        if split == 'test':
            res_filename = 'res_%s_[%s].txt' % (params['model'], time.time())
            res_filepath = os.path.join(params['test_res_dir'], res_filename)
            res = open(res_filepath, 'w')
            inverse_tag_dict = {ds.data['tag_set'][k]:k for k in ds.data['tag_set'].keys()}
            
        for i, ele in enumerate(ds.split[split]):
            Ys, cache = self.fwdPass(ele, params, predict_model=True)
            
            maxes = np.amax(Ys, axis=1, keepdims=True)
            e = np.exp(Ys - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            
            labels = np.array(ele['tags_rep'], dtype=int)
            
            if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)
            
            loss_cost = 0
            loss_cost += -np.sum(np.log(smooth_cost + probs[range(len(labels)), labels]))
            total_cost += loss_cost
            
            pred_words_indices = np.nanargmax(probs, axis=1)
            
            tokens = ele['raw_seq']
            real_tags = ele['tag_seq']
            for index, l in enumerate(labels):
                if pred_words_indices[index] == l: acc += 1
                
                if split == 'test':
                    res.write('%s %s %s %s\n' % (tokens[index], 'NA', real_tags[index], inverse_tag_dict[pred_words_indices[index]]))
            if split == 'test': res.write('\n')
            total += len(labels)
            
        total_cost /= len(ds.split[split])
        accuracy = 0 if total == 0 else float(acc)/total
        
        #print ("total_cost: %s, accuracy: %s" % (total_cost, accuracy))
        result = {'cost': total_cost, 'accuracy': accuracy}
        return result