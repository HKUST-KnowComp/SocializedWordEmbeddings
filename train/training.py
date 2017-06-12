import sys, os
import argparse

def train_swe(para_lambda,para_r,yelp_round):
	if os.path.isfile('./swe') == False: 
		command = 'gcc swe.c -o swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)
	
	lambda_str = str(para_lambda)
	lambda_index = lambda_str.index('.')
	lambda_str = lambda_str[0:lambda_index]+'p'+lambda_str[lambda_index+1:]

	r_str = str(para_r)
	r_index = r_str.index('.')
	r_str = r_str[0:r_index]+'p'+r_str[r_index+1:]

	swe_word_vec_file = '../data/swe_word_vec_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str, r_str)
	swe_user_vec_file = '../data/swe_user_vec_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str, r_str)
	swe_context_vec_file = '../data/swe_context_vec_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str, r_str)

	print 'Train Socialized Word Embeddings:\n'
	command = './swe -train ../data/swe_train_rd%d.txt \
-user ../data/user_file_rd%d.txt -user-graph ../data/user_graph_rd%d.txt \
-output %s -save-user %s \
-save-context %s -size 100 \
-window 5 -cbow 1 -hs 1 -negative 5 \
-lambda %.5f -r %.5f -threads 5 -iter 5 -sample 1e-4' %(yelp_round, yelp_round, yelp_round,
	swe_word_vec_file, swe_user_vec_file, swe_context_vec_file,
	para_lambda, para_r)
	print command
	os.system(command)


def train_w2v(yelp_round):
	if os.path.isfile('w2v') == False: 
		command = 'gcc w2v.c -o w2v -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)

	w2v_word_vec_file = '../data/w2v_word_vec_rd%d.txt' %(yelp_round)
	w2v_context_vec_file = '../data/w2v_context_vec_rd%d.txt' %(yelp_round)

	if os.path.isfile(w2v_word_vec_file) == True and os.path.isfile(w2v_context_vec_file) == True:
		print '%s and %s already exists. If you want to re-train word to vector, please delete it first.' %(w2v_word_vec_file, w2v_context_vec_file)
		return None

	print 'Train Word to Vector:\n'
	command = './w2v -train ../data/w2v_train_rd%d.txt \
-output %s \
-save-context %s -size 100 \
-window 5 -cbow 1 -hs 1 -negative 5 \
-threads 5 -iter 5 -sample 1e-4' %(yelp_round,
	w2v_word_vec_file, w2v_context_vec_file)
	print command
	os.system(command)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train Socialized Word Embeddings and Word to Vector')

	parser.add_argument('--para_lambda', default=None, type=float,
                        help='The trade off parameter between log-likelihood and regularization term')
	parser.add_argument('--para_r', default=None, type=float,
                        help="The constraint of the L2-norm of user vector")
	parser.add_argument('--yelp_round', default=9, type=int, choices={8,9},
                        help="The round number of yelp data")

	args = parser.parse_args()

	parser.print_help()

	train_swe(args.para_lambda,args.para_r,args.yelp_round)
	train_w2v(args.yelp_round)
