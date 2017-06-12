import sys, os
import argparse

def ppl_test(lambda_str,r_str,yelp_round):
	if os.path.isfile('./PPL_swe') == False: 
		command = 'gcc PPL_swe.c -o PPL_swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)
	if os.path.isfile('./PPL_w2v') == False: 
		command = 'gcc PPL_w2v.c -o PPL_w2v -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)

	test_file = '../data/PPL_test_rd%d.txt' %(yelp_round)
	
	word_vec_file = '../data/swe_word_vec_rd%d_l%s_r%s.txt' %(yelp_round,lambda_str,r_str)
	context_vec_file = '../data/swe_context_vec_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str,r_str)
	user_vec_file = '../data/swe_user_vec_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str,r_str)
	output_sentence_file = '../data/ppl_test_sentence_swe_rd%d_l%s_r%s.txt'%(yelp_round, lambda_str,r_str)
	
	print '\nPerplexity of Socialized Word Embeddings on Dev Set'
	command = './PPL_swe -user %s -word-vec %s -test %s -context-vec %s -sentence %s' %(user_vec_file, word_vec_file, test_file, context_vec_file, output_sentence_file)
	print command
	os.system(command)

	word_vec_file = '../data/w2v_word_vec_rd%d.txt' %(yelp_round)
	context_vec_file = '../data/w2v_context_vec_rd%d.txt' %(yelp_round)
	output_sentence_file = '../data/ppl_test_sentence_w2v_rd%d.txt'%(yelp_round)

	print '\nPerplexity of Word to Vector on Dev Set'
	command = './PPL_w2v -word-vec %s -test %s -context-vec %s -sentence %s' %( word_vec_file, test_file, context_vec_file,output_sentence_file)
	print command
	os.system(command)

def ppl_dev(lambda_str,r_str,yelp_round):
	if os.path.isfile('./PPL_swe') == False: 
		command = 'gcc PPL_swe.c -o PPL_swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)
	if os.path.isfile('./PPL_w2v') == False: 
		command = 'gcc PPL_w2v.c -o PPL_w2v -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)

	dev_file = '../data/PPL_dev_rd%d.txt' %(yelp_round)
	
	word_vec_file = '../data/swe_word_vec_rd%d_l%s_r%s.txt' %(yelp_round,lambda_str,r_str)
	context_vec_file = '../data/swe_context_vec_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str,r_str)
	user_vec_file = '../data/swe_user_vec_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str,r_str)
	output_sentence_file = '../data/ppl_dev_sentence_swe_rd%d_l%s_r%s.txt'%(yelp_round, lambda_str,r_str)
	
	print '\nPerplexity of Socialized Word Embeddings on Test Set'
	command = './PPL_swe -user %s -word-vec %s -test %s -context-vec %s -sentence %s' %(user_vec_file, word_vec_file, dev_file, context_vec_file, output_sentence_file)
	print command
	os.system(command)

	word_vec_file = '../data/w2v_word_vec_rd%d.txt' %(yelp_round)
	context_vec_file = '../data/w2v_context_vec_rd%d.txt' %(yelp_round)
	output_sentence_file = '../data/ppl_dev_sentence_w2v_rd%d.txt'%(yelp_round)

	print '\nPerplexity of Word to Vector on Test set'
	command = './PPL_w2v -word-vec %s -test %s -context-vec %s -sentence %s' %( word_vec_file, dev_file, context_vec_file, output_sentence_file)
	print command
	os.system(command)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Perplexity')

	parser.add_argument('--para_lambda', default=None, type=float,
                        help='The trade off parameter between log-likelihood and regularization term')
	parser.add_argument('--para_r', default=None, type=float,
                        help='The constraint of L2-norm of the user vector')
	parser.add_argument('--yelp_round', default=9, type=int, choices={8,9},
                        help='The round number of yelp data')

	args = parser.parse_args()

	parser.print_help()

	lambda_str = str(args.para_lambda)
	lambda_index = lambda_str.index('.')
	lambda_str = lambda_str[0:lambda_index]+'p'+lambda_str[lambda_index+1:]

	r_str = str(args.para_r)
	r_index = r_str.index('.')
	r_str = r_str[0:r_index]+'p'+r_str[r_index+1:]

	ppl_dev(lambda_str,r_str,args.yelp_round)
	ppl_test(lambda_str,r_str,args.yelp_round)
	
