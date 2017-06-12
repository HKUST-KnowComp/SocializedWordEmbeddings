import os, sys, random, argparse, time, math, gzip
import cPickle as pickle
from collections import Counter

import numpy as np
import theano
import theano.tensor as T

from nn import get_activation_by_name, create_optimization_updates, softmax, tanh
from nn import Layer, EmbeddingLayer, LSTM, Dropout, apply_dropout, CNN, IterAttentionLayer
from utils import say, load_embedding_iterator

np.set_printoptions(precision=3)

def read_corpus_doc(path, hier = 1):
    with open(path) as fin:
	lines = fin.readlines()

    rng = np.random.RandomState(66666)
    tmp = lines
    
    lines = []
    for x in tmp:
	if len(x.strip().split('\t\t')) == 3:
	    lines.append(x)
    segs = [x.strip().split('\t\t') for x in lines]
    
    if not hier:
	tmp_x = [ x[2].replace("#", " ") for x in segs]
    else:
    	tmp_x = [ x[2].split('#')  for x in segs]
    corpus_x = map(lambda x: map(lambda sent: sent.strip().split(), x), tmp_x)
    corpus_y = [ int(x[1]) - 1 for x in segs]
    corpus_user = [x[0] for x in segs]
    return corpus_x, corpus_y, corpus_user


def read_corpus(path):
    with open(path) as fin:
        lines = fin.readlines()
    
    segs = [x.strip().split('\t\t') for x in lines]
    corpus_x = [ x[3] for x in segs]
    corpus_y = [ int(x[2]) - 1 for x in segs]
    corpus_user = [x[0] for x in segs]
    corpus_prod = [x[1] for x in segs]
    return corpus_x, corpus_y

def create_one_batch(ids, x, y):
    batch_x = np.column_stack( [ x[i] for i in ids ] )
    batch_y = np.array( [ y[i] for i in ids ] )
    return batch_x, batch_y

def create_one_batch_doc(ids, x, y, u):
    max_len = 0
    for iid in ids:
	for sent in x[iid]:
	    if max_len < len(sent):
		max_len = len(sent)
    batch_x = map(lambda iid: np.asarray(map(lambda sent: sent + [-1] * (max_len - len(sent)), x[iid]), dtype = np.int32).T, ids)
    batch_w_mask = map(lambda iid: np.asarray(map(lambda sent: len(sent) * [1] + [0] * (max_len - len(sent)), x[iid]), dtype = np.float32).T, ids)
    batch_w_len = map(lambda iid: np.asarray(map(lambda sent: len(sent), x[iid]), dtype = np.float32) + np.float32(1e-4), ids)

    batch_x = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 1), batch_x)
    batch_w_mask = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 1), batch_w_mask)
    batch_w_len = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 0), batch_w_len)
    batch_y = np.array( [ y[i] for i in ids ] )
    batch_u = np.array( [ u[i] for i in ids ])
    return batch_x, batch_y, batch_u, batch_w_mask, batch_w_len, max_len

def create_batches_doc(perm, x, y, u, batch_size):
    lst = sorted(perm, key=lambda i: len(x[i]))
    batches_x = [ ]
    batches_w_masks = []
    batches_w_lens = []
    batches_sent_maxlen = []
    batches_sent_num = []
    batches_y = []
    batches_u = []
    size = batch_size
    ids = [ lst[0] ]
    for i in lst[1:]:
        if len(ids) < size and len(x[i]) == len(x[ids[0]]):
            ids.append(i)
        else:
            bx, by, bu, bm, bl, ml = create_one_batch_doc(ids, x, y, u)
            batches_x.append(bx)
            batches_y.append(by)
	    batches_u.append(bu)
	    batches_w_masks.append(bm)
	    batches_w_lens.append(bl)
	    batches_sent_num.append(len(x[ids[0]]))
	    batches_sent_maxlen.append(ml)
            ids = [ i ]
    bx, by, bu, bm, bl, ml = create_one_batch_doc(ids, x, y, u)
    batches_x.append(bx)
    batches_y.append(by)
    batches_u.append(bu)
    batches_w_masks.append(bm)
    batches_w_lens.append(bl)
    batches_sent_num.append(len(x[ids[0]]))
    batches_sent_maxlen.append(ml)
    # shuffle batches
    batch_perm = range(len(batches_x))
    random.shuffle(batch_perm)
    batches_x = [ batches_x[i] for i in batch_perm ]
    batches_y = [ batches_y[i] for i in batch_perm ]
    batches_u = [ batches_u[i] for i in batch_perm] 
    batches_w_masks = [ batches_w_masks[i] for i in batch_perm ]
    batches_w_lens = [ batches_w_lens[i] for i in batch_perm ]
    batches_sent_maxlen = [ batches_sent_maxlen[i] for i in batch_perm ]
    batches_sent_num = [ batches_sent_num[i] for i in batch_perm ]

    return batches_x, batches_y, batches_u, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num
    

# shuffle training examples and create mini-batches
def create_batches(perm, x, y, batch_size):

    # sort sequences based on their length
    # permutation is necessary if we want different batches every epoch
    lst = sorted(perm, key=lambda i: len(x[i]))

    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    ids = [ lst[0] ]
    for i in lst[1:]:
        if len(ids) < size and len(x[i]) == len(x[ids[0]]):
            ids.append(i)
        else:
            bx, by = create_one_batch(ids, x, y)
            batches_x.append(bx)
            batches_y.append(by)
            ids = [ i ]
    bx, by = create_one_batch(ids, x, y)
    batches_x.append(bx)
    batches_y.append(by)

    # shuffle batches
    batch_perm = range(len(batches_x))
    random.shuffle(batch_perm)
    batches_x = [ batches_x[i] for i in batch_perm ]
    batches_y = [ batches_y[i] for i in batch_perm ]
    return batches_x, batches_y

class Model:
    def __init__(self, args, embedding_layer, nclasses, user_embedding_layer = None):
        self.args = args
        self.embedding_layer = embedding_layer
	self.user_embedding_layer = user_embedding_layer
        self.nclasses = nclasses

    def ready(self):
        args = self.args
        embedding_layer = self.embedding_layer
        user_embedding_layer = self.user_embedding_layer
	self.n_hidden = args.hidden_dim
        self.n_in = embedding_layer.n_d
        dropout = self.dropout = theano.shared(
                np.float64(args.dropout_rate).astype(theano.config.floatX)
            )

        # x is length * batch_size
        # y is batch_size
        self.x = T.imatrix('x')
	self.w_masks = T.fmatrix('mask')
	self.w_lens = T.fvector('lens')
	self.s_ml = T.iscalar('sent_maxlen')
	self.s_num = T.iscalar('sent_num')
        self.y = T.ivector('y')
	self.usr = T.ivector('users')

        x = self.x
        y = self.y
	usr = self.usr
	w_masks = self.w_masks
	w_lens = self.w_lens
	s_ml = self.s_ml
	s_num = self.s_num
        n_hidden = self.n_hidden
        n_emb = n_in = self.n_in

	layers = self.layers = [ ]
	
	slicesu = user_embedding_layer.forward(usr)
        slices  = embedding_layer.forward(x.ravel())
        self.slices = slices #### important for updating word embeddings

        # 3-d tensor, len * batch_size * n_in
        slices = slices.reshape( (x.shape[0], x.shape[1], n_in) )
        
	pooling = args.pooling
        args.fix_emb = True
	prev_output = slices
        prev_output = apply_dropout(prev_output, dropout, v2=True)
        size = 0
	
	n_hidden_t = n_hidden
	if args.direction == "bi":
	    n_hidden_t = 2 * n_hidden
        
	softmax_inputs = [ ]
        activation = get_activation_by_name(args.act)
            
	if args.layer.lower() == "lstm":
            layer = LSTM(n_in = n_in,
            		n_out = n_hidden_t,
			direction = args.direction
                    )
	elif args.layer.lower() == "cnn":
	    layer = CNN(n_in = n_in,
                        n_out = n_hidden_t,
                        activation = activation,
                        order = args.order,
                        direction = args.direction
                    )
	else:
            raise Exception("unknown layer type: {}".format(args.layer))

        layers.append(layer)
        prev_output = layer.forward_all(prev_output, masks = w_masks)
        prev_output = apply_dropout(prev_output, dropout)

        # final feature representation is the concatenation of all extraction layers
        if args.user_atten:
	    layer = IterAttentionLayer(
                    n_in = n_emb,
                    n_out = n_hidden_t
        	)
            layers.append(layer)
	    if args.user_atten_base:
		slicesu = None
	    softmax_input = layers[-1].multi_hop_forward(prev_output, user_embs = slicesu, isWord = True, masks = w_masks)
	else:
	    if pooling:
	    	softmax_input = T.sum(prev_output, axis = 0) / w_lens.dimshuffle(0, 'x')
            else:
	    	ind = T.cast(w_lens - T.ones_like(w_lens), 'int32')
            	softmax_input = prev_output[T.arange(ind.shape[0]), ind]

        softmax_input = apply_dropout(softmax_input, dropout, v2=True)
	
        n_in = n_hidden_t
	size = 0
	softmax_inputs = [ ]
	[sentlen, emblen] = T.shape(softmax_input)
	prev_output = softmax_input.reshape((sentlen / s_num, s_num, emblen)).dimshuffle(1, 0, 2)
        if args.layer.lower() == "lstm":
            layer = LSTM(n_in = n_in,
                         n_out = n_hidden_t,
			 direction = args.direction
                    )
	elif args.layer.lower() == "cnn":
	    layer = CNN(n_in = n_in,
                        n_out = n_hidden_t,
                        activation = activation,
                        order = args.order,
                    )
        else:
    	    raise Exception("unknown layer type: {}".format(args.layer))

        layers.append(layer)
        prev_output = layer.forward_all(prev_output)
        prev_output = apply_dropout(prev_output, dropout)

        if args.user_atten:
            layer = IterAttentionLayer(
                    n_in = n_emb,
                    n_out = n_hidden_t
                )
            layers.append(layer)
	    
	    if args.user_atten_base:
		slicesu = None
            softmax_input = layers[-1].multi_hop_forward(prev_output, user_embs = slicesu, isWord = False)
	else:
	    if pooling:
            	softmax_input =  T.sum(prev_output, axis = 0) / T.cast(s_num, 'float32')
	    else:
            	softmax_input = prev_output[-1]
        softmax_input = apply_dropout(softmax_input, dropout, v2=True)
	    
	size = n_hidden_t
        layers.append( Layer(
                n_in = size,
                n_out = self.nclasses,
                activation = softmax,
                has_bias = False
        ) )
	args.fix_emb = True
	if args.fix_emb == True:
            for l,i in zip(layers, range(len(layers))):
            	say("layer {}: n_in={}\tn_out={}\n".format(
                    i, l.n_in, l.n_out
            	))
	else:
            for l,i in zip(layers[1:], range(len(layers[1:]))):
                say("layer {}: n_in={}\tn_out={}\n".format(
                    i, l.n_in, l.n_out
                ))

        # unnormalized score of y given x
        self.p_y_given_x = layers[-1].forward(softmax_input)
        self.pred = T.argmax(self.p_y_given_x, axis=1)
        self.nll_loss = T.mean( T.nnet.categorical_crossentropy(
                                    self.p_y_given_x,
                                    y
                            ))

        # adding regularizations
        self.l2_sqr = None
        self.params = [ ]
        for layer in layers:
            self.params += layer.params
        for p in self.params:
            if self.l2_sqr is None:
                self.l2_sqr = args.l2_reg * T.sum(p**2)
            else:
                self.l2_sqr += args.l2_reg * T.sum(p**2)

        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                        for x in self.params)
        say("total # parameters: {}\n".format(nparams))

    def save_model(self, path, args):
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.params ], args, self.nclasses),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            param_values, args, nclasses = pickle.load(fin)
	
	args.load = self.args.load
        self.args = args
        self.nclasses = nclasses
        self.ready()
        for x,v in zip(self.params, param_values):
            x.set_value(v)

    def eval_accuracy(self, preds, golds):
        fine = sum([ sum(p == y) for p,y in zip(preds, golds) ]) + 0.0
        fine_tot = sum( [ len(y) for y in golds ] )
        return fine/fine_tot

    def eval_accuracy_mse(self, preds, golds):
	error = sum([ (p - y) * (p - y) for ps, ys in zip(preds, golds) for p, y in zip(ps, ys)]) + 0.0
	fine_tot = sum( [ len(y) for y in golds ] )
	return error / fine_tot

    def train(self, train, dev, test):
        args = self.args
        trainx, trainy, trainu = train
        batch_size = args.batch

        if dev:
	    dev_batches_x, dev_batches_y, dev_batches_u, dev_batches_w_masks, dev_batches_w_lens, dev_batches_sent_maxlen, dev_batches_sent_num = create_batches_doc(
		    range(len(dev[0])),
		    dev[0],
		    dev[1],
		    dev[2],
		    128
		)

        if test:
	    test_batches_x, test_batches_y, test_batches_u, test_batches_w_masks, test_batches_w_lens, test_batches_sent_maxlen, test_batches_sent_num = create_batches_doc(
		    range(len(test[0])),
		    test[0],
		    test[1],
		    test[2],
		    128
		)

        cost = self.nll_loss + self.l2_sqr

        updates, lr, gnorm = create_optimization_updates(
                cost = cost,
                params = self.params,
                lr = args.learning_rate,
                method = args.learning
            )[:3]

        train_model = theano.function(
             	inputs = [self.x, self.y, self.usr, self.w_masks, self.w_lens, self.s_ml, self.s_num],
             	outputs = [ cost, gnorm ],
             	updates = updates,
             	allow_input_downcast = True
        	)

        eval_acc = theano.function(
             	inputs = [self.x, self.usr, self.w_masks, self.w_lens, self.s_ml, self.s_num],
             	outputs = self.pred,
             	allow_input_downcast = True
        	)

        unchanged = 0
        best_dev = 0.0
        dropout_prob = np.float64(args.dropout_rate).astype(theano.config.floatX)

        start_time = time.time()
        eval_period = args.eval_period

        perm = range(len(trainx))

        say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")
        if args.load:
	    self.dropout.set_value(0.0)
	    preds = [ eval_acc(x, u, wm, wl, sm, sn) for x, u, wm, wl, sm, sn in zip(dev_batches_x, dev_batches_u, dev_batches_w_masks, dev_batches_w_lens, dev_batches_sent_maxlen, dev_batches_sent_num)]
	    best_dev = self.eval_accuracy(preds, dev_batches_y)
	    mse_dev = self.eval_accuracy_mse(preds, dev_batches_y)
            say("\tdev mse = %.4f\taccuracy=%.4f\tbest=%.4f\n" % (
		mse_dev,
		best_dev,
		best_dev
	    ))
	    preds = [ eval_acc(x, u, wm, wl, sm, sn) for x, u, wm, wl, sm, sn in zip(test_batches_x, test_batches_u, test_batches_w_masks, test_batches_w_lens, test_batches_sent_maxlen, test_batches_sent_num)]
            nowf_test = self.eval_accuracy(preds, test_batches_y)
	    mse_test = self.eval_accuracy_mse(preds, test_batches_y)
            say("\tdev mse = %.4f\ttest accuracy=%.4f\n" % (
		 mse_test,
                 nowf_test
            ))

	
	test_a = 0.0
	for epoch in xrange(args.max_epochs):
            unchanged += 1
            if unchanged > 20: return
            train_loss = 0.0

            random.shuffle(perm)
	    
	    batches_x, batches_y, batches_u, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num = create_batches_doc(perm, trainx, trainy, trainu, batch_size)
            N = len(batches_x)
            for i in xrange(N):
		self.dropout.set_value(dropout_prob)
                if (i + 1) % 100 == 0:
                    sys.stdout.write("\r%d" % i)
                    sys.stdout.flush()

                x = batches_x[i]
                y = batches_y[i]
		
		va, grad_norm = train_model(x, y, batches_u[i], batches_w_masks[i], batches_w_lens[i], batches_sent_maxlen[i], batches_sent_num[i])
                train_loss += va

                # debug
                if math.isnan(va):
                    print ""
                    print i-1, i
                    print x
                    print y
		    #print batches_w_masks[i]
		    #print batches_w_lens[i]
		    print batches_sent_maxlen[i]
		    print batches_sent_num[i]
                    return

                if (i == N-1) or (eval_period > 0 and (i+1) % eval_period == 0):
                    self.dropout.set_value(0.0)

                    say( "\n" )
                    say( "Epoch %.3f\tloss=%.4f\t|g|=%s  [%.2fm]\n" % (
                            epoch + (i+1)/(N+0.0),
                            train_loss / (i+1),
                            float(grad_norm),
                            (time.time()-start_time) / 60.0
                    ))
                    say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")

                    if dev:
			preds = [ eval_acc(x, u, wm, wl, sm, sn) for x, u, wm, wl, sm, sn in zip(dev_batches_x, dev_batches_u, dev_batches_w_masks, dev_batches_w_lens, dev_batches_sent_maxlen, dev_batches_sent_num)]
                        nowf_dev = self.eval_accuracy(preds, dev_batches_y)
                        if nowf_dev > best_dev:
                            unchanged = 0
                            best_dev = nowf_dev
                            if args.save:
                                self.save_model(args.save, args)

                        say("\tdev accuracy=%.4f\tbest=%.4f\n" % (
                                nowf_dev,
                                best_dev
                        ))
			say("\ttest current_accuracy=%.4f\n" % (
                                test_a
                        ))
                        if args.test and nowf_dev == best_dev:
			    preds = [ eval_acc(x, u, wm, wl, sm, sn) for x, u, wm, wl, sm, sn in zip(test_batches_x, test_batches_u, test_batches_w_masks, test_batches_w_lens, test_batches_sent_maxlen, test_batches_sent_num)]
                            nowf_test = self.eval_accuracy(preds, test_batches_y)
                            say("\ttest accuracy=%.4f\n" % (
                                    nowf_test,
                            ))
			    test_a = nowf_test

                        if best_dev > nowf_dev + 0.5:
                            return

                    self.dropout.set_value(dropout_prob)

                    start_time = time.time()

    def evaluate_batches(self, batches_x, batches_y, eval_function):
        preds = [ eval_function(x) for x in batches_x ]
        return self.eval_accuracy(preds, batches_y)

    def evaluate_batches_doc(self, batches_x, batches_y, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num, eval_function):
	preds = [eval_function(x, wm, wl, sm, sn) for x, wm, wl, sm, sn in zip(batches_x, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num)]
	return self.eval_accuracy(preds, batches_y)

    def evaluate_set(self, data_x, data_y):

        args = self.args

        
        eval_acc = theano.function(
                inputs = [self.x, self.w_masks, self.w_lens, self.s_ml, self.s_num],
                outputs = self.pred,
                allow_input_downcast = True
                )

        # create batches by grouping sentences of the same length together
	
	batches_x, batches_y, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num = create_batches_doc(
                    range(len(data_x[0])),
                    data_x,
                    data_y,
                    args.batch
                )

        # evaluate on the data set
        dropout_prob = np.float64(args.dropout_rate).astype(theano.config.floatX)
        self.dropout.set_value(0.0)
	accuracy = self.evaluate_batches_doc(batches_x, batches_y, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num, eval_acc)

        self.dropout.set_value(dropout_prob)
        return accuracy

def load_doc_corpus(embedding_layer,user_embedding_layer, fileIn, hieri = 1):
    corpus_x, corpus_y, corpus_usr = read_corpus_doc(fileIn, hier = hieri)
    corpus_x = map(lambda x: map(lambda sent: embedding_layer.map_to_ids(sent).tolist(), x), corpus_x)
    corpus_usr = user_embedding_layer.map_to_ids(corpus_usr).tolist()
    return corpus_x, corpus_y, corpus_usr

def load_sent_corpus(embedding_layer, fileIn):
    corpus_x, corpus_y = read_corpus(fileIn)
    corpus_x = [ embedding_layer.map_to_ids(x) for x in corpus_x ]
    return corpus_x, corpus_y

def load_list(lis, fileIn):
    with open(fileIn, 'r') as fin:
	for line in fin.readlines():
	    lis.append(line.strip())
    return lis

def main(args):
    args.fix_emb = True
    print args

    model = None

    assert args.embedding, "Pre-trained word embeddings required."
    
    embedding_layer = EmbeddingLayer(
                n_d = args.hidden_dim,
                vocab = [ "<unk>" ],
                embs = load_embedding_iterator(args.embedding),
		fix_init_embs = args.fix_emb
            )
	
    user_embedding_layer = None

    fix_embs = True
    vocab_lis = ["<unk>"]
    
    if args.user_embs:
	user_embedding_layer= EmbeddingLayer(
                n_d = args.hidden_dim,
                vocab =vocab_lis,
                embs = load_embedding_iterator(args.user_embs),
                fix_init_embs = fix_embs,
		norm = args.norm_user_embs
            )
	n_d = user_embedding_layer.n_d 
    else:
	user_embedding_layer= EmbeddingLayer(
            	n_d = n_d,
                vocab =vocab_lis,
                embs = None,
                fix_init_embs = fix_embs,
		norm = args.norm_user_embs
            )

    if args.train:
	train_x, train_y, train_usr = load_doc_corpus(embedding_layer, user_embedding_layer, args.train, hieri = args.hierarchical)

    if args.dev:
	dev_x, dev_y, dev_usr = load_doc_corpus(embedding_layer, user_embedding_layer, args.dev, hieri = args.hierarchical)

    if args.test:
	test_x, test_y, test_usr = load_doc_corpus(embedding_layer, user_embedding_layer, args.test, hieri = args.hierarchical)

    if args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    nclasses = max(train_y) + 1,
		    user_embedding_layer = user_embedding_layer,
            )
	if args.load:
	    print 'loading model...'
	    model.load_model(args.load)
        else:
	    model.ready()
	
	print 'training...'
        model.train(
                (train_x, train_y, train_usr),
                (dev_x, dev_y, dev_usr) if args.dev else None,
                (test_x, test_y, test_usr) if args.test else None,
            )

    if args.load and args.test and not args.train:
        # model.args and model.nclasses will be loaded from file
        model = Model(
                    args = None,
                    embedding_layer = embedding_layer,
                    nclasses = -1
            )
        model.load_model(args.load)
        accuracy = model.evaluate_set(test_x, test_y)
        print accuracy


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train",
            type = str,
            default = "",
            help = "path to training data"
        )
    argparser.add_argument("--dev",
            type = str,
            default = "",
            help = "path to development data"
        )
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data"
        )
    argparser.add_argument("--user_embs",
	    type = str,
	    default = "",
	    help = "path to file for user embeddings"
	)
    argparser.add_argument("--direction",
	    type = str,
	    default = "bi",
	    help = "direction: si, bi; only work for lstm"
	)
    argparser.add_argument("--fix_emb",
	    type = bool,
	    default = False,
	    help = "fine tune embedding: True or False"
	)
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 25,
            help = "hidden dimensions"
        )
    argparser.add_argument("--decay",
            type = float,
            default = 0.5,
            help = "the decay factor of StrCNN layer"
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adadelta",
            help = "learning method (sgd, adagrad, adam, ...)"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = "0.001",
            help = "learning rate"
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--eval_period",
            type = int,
            default = 1000,
            help = "evaluate on dev every period"
        )
    argparser.add_argument("--dropout_rate",
            type = float,
            default = 0,
            help = "dropout probability"
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 0.00001
        )
    argparser.add_argument("--embedding",
            type = str,
            default = ""
        )
    argparser.add_argument("--batch",
            type = int,
            default = 64,
            help = "mini-batch size"
        )
    argparser.add_argument("--order",
            type = int,
            default = 2,
            help = "when the order is k, we use up tp k-grams"
        )
    argparser.add_argument("--act",
            type = str,
            default = "relu",
            help = "activation function (none, relu, tanh, etc.)"
        )
    argparser.add_argument("--layer",
            type = str,
            default = "lstm",
            help = "type of neural net (LSTM, RCNN, StrCNN)"
        )
    argparser.add_argument("--mode",
            type = int,
            default = 1
        )
    argparser.add_argument("--save",
            type = str,
            default = "",
            help = "save model to this file"
        )
    argparser.add_argument("--load",
            type = str,
            default = "",
            help = "load model from this file"
        )
    argparser.add_argument("--pooling",
            type = int,
            default = 1,
            help = "whether to use mean pooling or take the last vector"
        )
    argparser.add_argument("--user_atten",
	    type = int,
	    default = 1,
	    help = "whether to use user attention "
	)
    argparser.add_argument("--user_atten_base",
	    type = int,
	    default = 0,
	    help = "baseline for user attention model"
	)
    argparser.add_argument("--hierarchical",
            type = int,
            default = 1,
            help = "whether to learn model in a hierarchical architecture."
        )
    argparser.add_argument("--norm_user_embs",
	    type = float,
	    default = -1,
	    help = "normilize the user embeddings"
	)
    args = argparser.parse_args()
    main(args)

