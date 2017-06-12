THEANO_FLAGS="floatX=float32,device=gpu0,cuda.root=/usr/local/cuda,on_unused_input=ignore,optimizer=fast_compile"  python dc.py --embedding embs/embs.words.sample --train data/train.sample   --dev data/dev.sample  --test data/test.sample --save model/just4test --user_embs embs/embs.users.sample  --layer lstm --user_atten 1 --user_atten_base 1 
# --learning sgd
