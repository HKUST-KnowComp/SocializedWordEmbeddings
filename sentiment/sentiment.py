import sys, os
import argparse
import numpy as np
from prettytable import PrettyTable


def SVM_format_w2v(yelp_round,word_vec_w2v):

	# w2v: get SVM format on ../data/SVM_train_one_fifth_rd%d.txt
	command = './get_SVM_format_w2v -input ../data/SVM_train_one_fifth_rd%d.txt \
	-word-vec %s \
	-output ../data/SVM_format_train_w2v_rd%d.txt ' %( 
	yelp_round,
	word_vec_w2v,
	yelp_round)
	print command
	os.system(command)

	# w2v: get SVM format on ../data/SVM_dev_one_fifth_rd%d.txt
	command = './get_SVM_format_w2v -input ../data/SVM_dev_one_fifth_rd%d.txt \
	-word-vec %s \
	-output ../data/SVM_format_dev_w2v_rd%d.txt ' %( 
	yelp_round,
	word_vec_w2v,
	yelp_round)
	print command
	os.system(command)

	# w2v: get SVM format on ../data/SVM_test_one_fifth_rd%d.txt
	command = './get_SVM_format_w2v -input ../data/SVM_test_one_fifth_rd%d.txt \
	-word-vec %s \
	-output ../data/SVM_format_test_w2v_rd%d.txt ' %( 
	yelp_round,
	word_vec_w2v,
	yelp_round)
	print command
	os.system(command)

def SVM_format_swe(yelp_round,word_vec, user_vec, lambda_str, r_str):
	
	# swe: get SVM format on ../data/SVM_train_one_fifth_rd%d.txt
	command = './get_SVM_format_swe -input ../data/SVM_train_one_fifth_rd%d.txt \
	-word-vec %s \
	-user-vec %s \
	-output ../data/SVM_format_train_swe_rd%d_l%s_r%s.txt' %(
	yelp_round,
	word_vec, 
	user_vec, 
	yelp_round, lambda_str, r_str)
	print command
	os.system(command)
	
	# swe: get SVM format on ../data/SVM_dev_one_fifth_rd%d.txt
	command = './get_SVM_format_swe -input ../data/SVM_dev_one_fifth_rd%d.txt \
	-word-vec %s \
	-user-vec %s \
	-output ../data/SVM_format_dev_swe_rd%d_l%s_r%s.txt' %(
	yelp_round,
	word_vec, 
	user_vec, 
	yelp_round, lambda_str, r_str)
	print command
	os.system(command)

	# swe: get SVM format on ../data/SVM_test_one_fifth_rd%d.txt
	command = './get_SVM_format_swe -input ../data/SVM_test_one_fifth_rd%d.txt \
	-word-vec %s \
	-user-vec %s \
	-output ../data/SVM_format_test_swe_rd%d_l%s_r%s.txt' %(
	yelp_round,
	word_vec, 
	user_vec, 
	yelp_round, lambda_str, r_str)
	print command
	os.system(command)

def tune_para_SVM_swe(word_vec,user_vec,lambda_str,r_str,yelp_round):
	##############################################
	# Tune Parameter C of SVM Classifier  
	##############################################
	if os.path.isfile('./get_SVM_format_swe') == False: 
		command = 'gcc get_SVM_format_swe.c -o get_SVM_format_swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)

	command = './get_SVM_format_swe -input ../data/SVM_dev_one_fifth_rd%d.txt -word-vec %s -user-vec %s -output SVM_format_dev_swe_rd%d_l%s_r%s.txt' %(
	yelp_round, word_vec, user_vec, yelp_round, lambda_str, r_str)
	print command
	os.system(command)
			
	path='./best_C_dev_swe_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str, r_str)
	command ='../liblinear/train -C -v 5 SVM_format_dev_swe_rd%d_l%s_r%s.txt |& tee -a %s' %(yelp_round, lambda_str, r_str, path)
	print command
	with open(path, 'wb') as fo:
		fo.write(command+'\n')
	os.system(command)

	command = 'rm SVM_format_dev_swe_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str, r_str)
	#print command
	os.system(command)

def tune_para_SVM_w2v(word_vec_w2v,yelp_round):
	##############################################
	# Tune Parameter C of SVM Classifier 
	##############################################
	if os.path.isfile('./get_SVM_format_w2v') == False: 
		command = 'gcc get_SVM_format_w2v.c -o get_SVM_format_w2v -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)

	path='./best_C_dev_w2v_rd%d.txt' %(yelp_round)

	if os.path.isfile(path) == True:
		print 'The file %s already exists. If you want to generate it again, please delete it first.'  %(path)
		return None

	command = './get_SVM_format_w2v -input ../data/SVM_dev_one_fifth_rd%d.txt -word-vec %s -output SVM_format_dev_w2v_rd%d.txt' %(
	yelp_round, word_vec_w2v, yelp_round)
	print command
	os.system(command)
			
	command ='../liblinear/train -C -v 5 SVM_format_dev_w2v_rd%d.txt |& tee -a %s' %(yelp_round, path)
	print command
	with open(path, 'wb') as fo:
		fo.write(command+'\n')
	os.system(command)

	command = 'rm SVM_format_dev_w2v_rd%d.txt' %(yelp_round)
	#print command
	os.system(command)

def test(yelp_round):

	# read best para_c_swe
	c_file='./best_C_dev_swe_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str, r_str)
	with open(c_file,"rb") as fin:
		for i, line in enumerate(fin):
			pass
		total = i
	with open(c_file,"rb") as fin:
		for i, line in enumerate(fin):
			if i == total:
				array_line = line.strip('\n').split()
				para_c_swe = float(array_line[3])

	# read best para_c_w2v
	c_file='./best_C_dev_w2v_rd%d.txt' %(yelp_round)
	with open(c_file,"rb") as fin:
		for i, line in enumerate(fin):
			pass
		total = i
	with open(c_file,"rb") as fin:
		for i, line in enumerate(fin):
			if i == total:
				array_line = line.strip('\n').split()
				para_c_w2v = float(array_line[3])

	# Check whether para_c_swe and para_c_w2v are reasonable
	# The best C may be very large. It will reach max number of iterations 
	# when doing classification using liblinear.
	# If the difference between para_c_swe and para_c_w2v are very large, 
	# let the large one to the same with the samll one.
	if para_c_swe >= 2 or para_c_w2v >= 2:
		min_c = min(para_c_swe,para_c_w2v,2)
		para_c_swe = min_c
		para_c_w2v = min_c

	train_file_swe = '../data/SVM_format_train_swe_rd%d_l%s_r%s.txt' %(yelp_round,lambda_str,r_str)
	test_file_swe = '../data/SVM_format_test_swe_rd%d_l%s_r%s.txt' %(yelp_round,lambda_str,r_str)

	command = '../liblinear/train -c %f %s \
../data/SVM_format_train_swe_rd%d_l%s_r%s.model ' %(para_c_swe, 
	train_file_swe, 
	yelp_round, lambda_str, r_str )
	print command
	os.system(command)

	command = '../liblinear/predict \
%s \
../data/SVM_format_train_swe_rd%d_l%s_r%s.model result.txt |& tee \
result_test_swe_rd%d_l%s_r%s.txt' %( 
	test_file_swe,
	yelp_round, lambda_str, r_str, 
	yelp_round, lambda_str, r_str)
	print command
	os.system(command)

	train_file_w2v = '../data/SVM_format_train_w2v_rd%d.txt' %(yelp_round)
	test_file_w2v = '../data/SVM_format_test_w2v_rd%d.txt' %(yelp_round)

	command = '../liblinear/train -c %f %s \
../data/SVM_format_train_w2v_rd%d.model' %(para_c_w2v, 
	train_file_w2v,
	yelp_round)
	print command
	os.system(command)

	command = '../liblinear/predict \
%s \
../data/SVM_format_train_w2v_rd%d.model result.txt |& tee \
result_test_w2v_rd%d.txt' %( 
	test_file_w2v,
	yelp_round,
	yelp_round)
	print command
	os.system(command)

	result_file_swe = 'result_test_swe_rd%d_l%s_r%s.txt' %(yelp_round, lambda_str, r_str)
	result_file_w2v = 'result_test_w2v_rd%d.txt' %(yelp_round)
	acc_array = [0 for i in range(2)]
	with open(result_file_swe,"rb") as fin:
		for line in fin:
			line_array = line.split(' ')
			acc = line_array[-2]
			acc_array[0] = float(acc[:-1]) 
	command = 'rm %s' %(result_file_swe)
	#print command
	os.system(command)

	with open(result_file_w2v,"rb") as fin:
		for line in fin:
			line_array = line.split(' ')
			acc = line_array[-2]
			acc_array[1] = float(acc[:-1]) 
	command = 'rm %s' %(result_file_w2v)
	#print command
	os.system(command)

	x = PrettyTable()
	x.add_column('method',['SWE','W2V'])
	x.add_column('accuracy',acc_array)

	print (x)

	table_txt = x.get_string()
	with open('head_tail_output.txt','a+') as fo:
		title = '\nrd%d_l%s_r%s\n' %(yelp_round, lambda_str, r_str)
		fo.write(title)
		fo.write(table_txt)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Sentiment Classification on head and tail users using liblinear')

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

	word_vec = '../data/swe_word_vec_rd%d_l%s_r%s.txt' %(args.yelp_round, lambda_str, r_str)
	user_vec = '../data/swe_user_vec_rd%d_l%s_r%s.txt' %(args.yelp_round, lambda_str, r_str)
	word_vec_w2v = '../data/w2v_word_vec_rd%d.txt' %(args.yelp_round)

	SVM_format_swe(args.yelp_round, word_vec, user_vec, lambda_str, r_str)
	SVM_format_w2v(args.yelp_round, word_vec_w2v)
	
	tune_para_SVM_swe(word_vec, user_vec, lambda_str, r_str, args.yelp_round)
	tune_para_SVM_w2v(word_vec_w2v, args.yelp_round)
	
	test(args.yelp_round)