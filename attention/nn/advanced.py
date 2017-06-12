import numpy as np
import theano
import theano.tensor as T

from .initialization import random_init, create_shared
from .initialization import ReLU, tanh, linear, sigmoid
from .basic import Layer, RecurrentLayer

class IterAttentionLayer(Layer):
    def __init__(self, n_in, n_out):
	self.n_in = n_in
	self.n_out = n_out
	self.create_parameters()

    def create_parameters(self):
	n_in = self.n_in
	n_out = self.n_out
	self.W = create_shared(random_init((n_in, n_out)), name = "W")
	self.a = create_shared(random_init((n_out)), name = "a")
	self.V = create_shared(random_init((n_out, n_out)), name = "V")
	self.V1 = create_shared(random_init((n_out)), name = "V1")
	self.V2 = create_shared(random_init((n_out)), name = "V2")
	self.U = create_shared(random_init((n_in)), name = "U")
	self.lst_params = [self.W, self.a, self.V1, self.V2, self.V, self.U]

    def multi_hop_forward(self, prev_output, user_embs = None, isWord = True, hop = 1, masks = None):
	W = self.W
	V = self.V
	V1 = self.V1
	U = self.U
	a = self.a
	
	n_out = self.n_out
	
	doc_vecs = prev_output.dimshuffle(1, 0, 2)
	if isWord and user_embs:
	    doc_vecs = doc_vecs.reshape((user_embs.shape[0], doc_vecs.shape[0] / user_embs.shape[0], doc_vecs.shape[1], doc_vecs.shape[2]))
	
	s_Rs = []
	v_tmp_ =  T.dot(doc_vecs, V)  + a
	for i in range(hop):
	    v_tmp = v_tmp_
	    
	    if i == 0:
	    	if isWord:
		    if user_embs:
		    	v_tmp = v_tmp + T.dot(user_embs, W).dimshuffle(0, 'x', 'x', 1)
		    else:
		    	v_tmp = v_tmp + T.dot(U, W)
	    	else:
		    if user_embs:
                    	v_tmp = v_tmp + T.dot(user_embs, W).dimshuffle(0, 'x', 1)
                    else:
                    	v_tmp = v_tmp + T.dot(U, W)		
	    else:
		p_tmp = s_Rs[-1]
	
	    alpha = T.exp(T.dot(T.tanh(v_tmp), V1))
	    if masks is not None:
                if masks.dtype != theano.config.floatX:
                    masks = T.cast(masks, theano.config.floatX)
                if isWord and user_embs:
		    masks = masks.dimshuffle(1, 0)
		    masks = masks.reshape((user_embs.shape[0], masks.shape[0] / user_embs.shape[0], masks.shape[1]))
		    alpha = alpha * masks
		else:
		    alpha = alpha * masks.dimshuffle(1, 0)
		
	    if isWord and user_embs:
		alpha_S = T.sum(alpha, axis = 2)
		alpha = alpha / (alpha_S.dimshuffle(0, 1, 'x') + 1e-5)
		s_Rs.append(T.sum(doc_vecs * alpha.dimshuffle(0, 1, 2, 'x'), axis = 2))
	    else:
		alpha_S = T.sum(alpha, axis = 1)
		alpha = alpha / (alpha_S.dimshuffle(0, 'x') + 1e-5)
		s_Rs.append(T.sum(doc_vecs * alpha.dimshuffle(0, 1, 'x'), axis = 1))

	result_vec = s_Rs[-1]
	if isWord and user_embs:
	    result_vec = result_vec.reshape((result_vec.shape[0] * result_vec.shape[1], result_vec.shape[2]))
	return result_vec
    
    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())

class CNN(Layer):
    def __init__(self, n_in, n_out, activation=tanh,
            order=1,  clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients

        internal_layers = self.internal_layers = [ ]
        for i in range(order):
            input_layer = Layer(n_in, n_out, linear, has_bias=False, \
                    clip_gradients=clip_gradients)
            internal_layers.append(input_layer)

        self.bias = create_shared(random_init((n_out,)), name="bias")

    def forward(self, x, mask, hc):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t =  in_i_t
            else:
                c_i_t =  in_i_t + c_im1_tm1
            
	    lst.append(T.cast(c_i_t * mask.dimshuffle(0, 'x'), 'float32'))
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        h_t = activation(c_i_t + self.bias)
        lst.append(T.cast(h_t * mask.dimshuffle(0, 'x'), 'float32'))
        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

    def forward_all(self, x, masks = None, h0=None, return_c=False, direction = None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)

        if masks == None:
            masks = T.ones((x.shape[0], x.shape[1]), dtype = theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [x, masks],
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out*self.order:]
        else:
            return h[:,self.n_out*self.order:]


    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ] + [ self.bias ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        self.bias.set_value(param_list[-1].get_value())


