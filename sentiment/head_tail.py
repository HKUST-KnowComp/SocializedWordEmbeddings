import sys, os
import argparse
import numpy as np
from prettytable import PrettyTable

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

def get_head_tail_review(yelp_round):
	
	head_file = '../data/head_user_review_train_rd%d.txt' %( yelp_round )
	tail_file = '../data/tail_user_review_train_rd%d.txt' %( yelp_round )

	if os.path.isfile(head_file) == True and os.path.isfile(tail_file) == True: 
		return None

	dic = {} 
	dic_inverse = {}
	dic_head = {}
	dic_tail = {}
 
	user_count = 0 # how many users

	user_file = '../data/user_file_rd%d.txt' %(yelp_round)
	with open(user_file,"rb") as fin:
		for line in fin:
			user_id = line.strip('\n')
			if user_id not in dic.keys():
				dic[user_id] = user_count
				dic_inverse[user_count] = user_id
				user_count = user_count + 1
		total = user_count
	print "total %d user" %(total)

	#count how many reviews of each user in training file, ignore unknown user
	count_array = [0 for i in range(total)]
	train_file = '../data/SVM_train_one_fifth_rd%d.txt' %(yelp_round)
	total_review = 0
	with open(train_file,"rb") as f_train:
		for line in f_train:
			line_array = line.split(' ')
			if line_array[0] != 'unknown_user_id':
				count_array [ dic [ line_array[0] ] ] = count_array [ dic [ line_array[0] ]  ]+  1
				total_review = total_review + 1
		sorted_index = [b[0] for b in sorted(enumerate(count_array),key=lambda x:x[1]) ] # ascending order

	#get head & tail user dictionary
	head_cnt = 0
	tail_cnt = 0
	accumu_sum = 0
	for i in range(user_count):
		accumu_sum = accumu_sum + count_array[ sorted_index[i] ]
		if accumu_sum > total_review/2:
			lKey = dic_inverse[sorted_index[i]]
			dic_head[ lKey ] = head_cnt
			head_cnt = head_cnt + 1;
		else:
			lKey = dic_inverse[sorted_index[i]]
			dic_tail[ lKey ] = tail_cnt
			tail_cnt = tail_cnt + 1;

	print "%d head user:" %(head_cnt)
	print "%d tail user:" %(tail_cnt)

	#generate training file (in text format) that only contains head(tail) user 
	
	f_head = open(head_file,"wb") 
	f_tail = open(tail_file,'wb') 
	with open(train_file,"rb") as fin:
		for line in fin:
			line_array = line.split(' ')
			if line_array[0] in dic_head.keys():
				f_head.write(line)
			if line_array[0] in dic_tail.keys():
				f_tail.write(line)
	f_head.close()
	f_tail.close()

def SVM_format_w2v(yelp_round,word_vec_w2v):

	output_file = '../data/tail_user_SVM_format_train_w2v_rd%d.txt'  %(yelp_round)
	if os.path.isfile(output_file) == True: 
		print 
		return None

	# w2v: get SVM format on ../data/tail_user_review_train_rd%d.txt
	command = './get_SVM_format_w2v -input ../data/tail_user_review_train_rd%d.txt \
	-word-vec %s \
	-output ../data/tail_user_SVM_format_train_w2v_rd%d.txt ' %( 
	yelp_round,
	word_vec_w2v,
	yelp_round)
	print command
	os.system(command)

	# w2v: get SVM format on ../data/head_user_review_train_rd%d.txt
	command = './get_SVM_format_w2v -input ../data/head_user_review_train_rd%d.txt \
	-word-vec %s \
	-output ../data/head_user_SVM_format_train_w2v_rd%d.txt ' %( 
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

def SVM_format_swe(yelp_round,word_vec, user_vec, lambda_str, r_str):
	
	# swe: get SVM format on ../data/tail_user_review_train_rd%d.txt
	command = './get_SVM_format_swe -input ../data/tail_user_review_train_rd%d.txt \
	-word-vec %s \
	-user-vec %s \
	-output ../data/tail_user_SVM_format_train_swe_rd%d_l%s_r%s.txt' %( 
	yelp_round, 
	word_vec, 
	user_vec, 
	yelp_round, lambda_str, r_str)
	print command
	os.system(command)
	
	# swe: get SVM format on ../data/head_user_review_train_rd%d.txt
	command = './get_SVM_format_swe -input ../data/head_user_review_train_rd%d.txt \
	-word-vec %s \
	-user-vec %s \
	-output ../data/head_user_SVM_format_train_swe_rd%d_l%s_r%s.txt' %( 
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

def run_SVM_frac(yelp_round,lambda_str,r_str,factor):
	
	runs = 10 # The number of independent runs on a fractional training data
	
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
		
	# run SVM on fractional training data
	tail_swe = '../data/tail_user_SVM_format_train_swe_rd%d_l%s_r%s.txt' %( yelp_round, lambda_str, r_str)
	with open(tail_swe,"rb") as f_tail:
		for i, line in enumerate(f_tail):
			pass
		tail_total = i + 1
	print "tail_reviews %d" %(tail_total)

	head_swe = '../data/head_user_SVM_format_train_swe_rd%d_l%s_r%s.txt' %( yelp_round, lambda_str, r_str )
	with open(head_swe,"rb") as f_head:
		for i, line in enumerate(f_head):
			pass
		head_total = i + 1
	print "head_reviews %d" %(head_total)

	tail_w2v = '../data/tail_user_SVM_format_train_w2v_rd%d.txt' %( yelp_round)
	head_w2v = '../data/head_user_SVM_format_train_w2v_rd%d.txt' %( yelp_round)


	#############
	# tail user #
	#############
	random_tail_swe = '../data/tail_user_SVM_format_random_train_swe_rd%d_l%s_r%s_f%d.txt' %( yelp_round, lambda_str, r_str , int(factor*10)  )
	random_tail_w2v = '../data/tail_user_SVM_format_random_train_w2v_rd%d_l%s_r%s_f%d.txt' %( yelp_round, lambda_str, r_str , int(factor*10) )
	for it in range(runs):
		#select fractional training data (in text format) randomly
		random_array = np.random.permutation(tail_total)
		random_array = random_array[0:int(tail_total*factor)]
		with open (random_tail_swe,"wb") as fo:
			with open(tail_swe,"rb") as f_tail:
				for i, line in enumerate(f_tail):
					if i in random_array:
						fo.write(line)

		with open (random_tail_w2v,"wb") as fo:
			with open(tail_w2v,"rb") as f_tail:
				for i, line in enumerate(f_tail):
					if i in random_array:
						fo.write(line)


		command = '../liblinear/train -c %f %s \
../data/tail_user_SVM_format_random_train_swe_rd%d_l%s_r%s_f%d.model ' %(para_c_swe, 
	random_tail_swe, 
	yelp_round, lambda_str, r_str , int(factor*10) )
		print command
		os.system(command)

		command = '../liblinear/predict \
../data/SVM_format_dev_swe_rd%d_l%s_r%s.txt \
../data/tail_user_SVM_format_random_train_swe_rd%d_l%s_r%s_f%d.model result.txt |& tee \
result_tail_user_swe_rd%d_l%s_r%s_f%d_run%d.txt' %( 
		yelp_round, lambda_str, r_str,
		yelp_round, lambda_str, r_str , int(factor*10),
		yelp_round, lambda_str, r_str , int(factor*10), it )
		print command
		os.system(command)

		command = '../liblinear/train -c %f %s \
../data/tail_user_SVM_format_random_train_w2v_rd%d_l%s_r%s_f%d.model' %(para_c_w2v, 
	random_tail_w2v,
	yelp_round, lambda_str, r_str , int(factor*10) )
		print command
		os.system(command)

		command = '../liblinear/predict \
../data/SVM_format_dev_w2v_rd%d.txt \
../data/tail_user_SVM_format_random_train_w2v_rd%d_l%s_r%s_f%d.model result.txt |& tee \
result_tail_user_w2v_rd%d_l%s_r%s_f%d_run%d.txt' %( 
		yelp_round,
		yelp_round, lambda_str, r_str , int(factor*10),
		yelp_round, lambda_str, r_str , int(factor*10), it )
		print command
		os.system(command)

	command = 'rm %s' %(random_tail_swe)
	#print command
	os.system(command)

	command = 'rm ../data/tail_user_SVM_format_random_train_swe_rd%d_l%s_r%s_f%d.model' %(yelp_round, lambda_str, r_str , int(factor*10))
	#print command
	os.system(command)

	acc_array = [0 for i in range(runs)]
	for it in range(runs):
		result_file = 'result_tail_user_swe_rd%d_l%s_r%s_f%d_run%d.txt' %(yelp_round, lambda_str, r_str , int(factor*10), it)
		with open(result_file,"rb") as fin:
			for line in fin:
				line_array = line.split(' ')
				acc = line_array[-2]
				acc_array[it] = float(acc[:-1]) 
		command = 'rm %s' %(result_file)
		#print command
		os.system(command)

	# std = sqrt(mean(abs(x - x.mean())**2)).
	tail_swe_std = np.std(acc_array)
	tail_swe_mean = np.mean(acc_array)

	command = 'rm %s' %(random_tail_w2v)
	#print command
	os.system(command)

	command = 'rm ../data/tail_user_SVM_format_random_train_w2v_rd%d_l%s_r%s_f%d.model' %(yelp_round, lambda_str, r_str , int(factor*10))
	#print command
	os.system(command)

	acc_array = [0 for i in range(runs)]
	for it in range(runs):
		result_file = 'result_tail_user_w2v_rd%d_l%s_r%s_f%d_run%d.txt' %( yelp_round, lambda_str, r_str ,int(factor*10), it )
		with open(result_file,"rb") as fin:
			for line in fin:
				line_array = line.split(' ')
				acc = line_array[-2]
				acc_array[it] = float(acc[:-1]) 
		command = 'rm %s' %(result_file)
		#print command
		os.system(command)

	# std = sqrt(mean(abs(x - x.mean())**2)).
	tail_w2v_std = np.std(acc_array)
	tail_w2v_mean = np.mean(acc_array)


	#############
	# head user #
	#############

	random_head_swe = '../data/head_user_SVM_format_random_train_swe_rd%d_l%s_r%s_f%d.txt' %( yelp_round, lambda_str, r_str , int(factor*10)  )
	random_head_w2v = '../data/head_user_SVM_format_random_train_w2v_rd%d_l%s_r%s_f%d.txt' %( yelp_round, lambda_str, r_str , int(factor*10) )
	for it in range(runs):
		#select fractional training data (in text format) randomly
		random_array = np.random.permutation(head_total)
		random_array = random_array[0:int(head_total*factor)]
		with open (random_head_swe,"wb") as fo:
			with open(head_swe,"rb") as f_head:
				for i, line in enumerate(f_head):
					if i in random_array:
						fo.write(line)

		with open (random_head_w2v,"wb") as fo:
			with open(head_w2v,"rb") as f_head:
				for i, line in enumerate(f_head):
					if i in random_array:
						fo.write(line)


		command = '../liblinear/train -c %f %s \
../data/head_user_SVM_format_random_train_swe_rd%d_l%s_r%s_f%d.model ' %(para_c_swe, 
	random_head_swe, 
	yelp_round, lambda_str, r_str , int(factor*10) )
		print command
		os.system(command)

		command = '../liblinear/predict \
../data/SVM_format_dev_swe_rd%d_l%s_r%s.txt \
../data/head_user_SVM_format_random_train_swe_rd%d_l%s_r%s_f%d.model result.txt |& tee \
result_head_user_swe_rd%d_l%s_r%s_f%d_run%d.txt' %( 
	yelp_round, lambda_str, r_str,
	yelp_round, lambda_str, r_str , int(factor*10),
	yelp_round, lambda_str, r_str , int(factor*10), it )
		print command
		os.system(command)

		command = '../liblinear/train -c %f %s \
../data/head_user_SVM_format_random_train_w2v_rd%d_l%s_r%s_f%d.model' %(para_c_w2v, 
	random_head_w2v,
	yelp_round, lambda_str, r_str , int(factor*10) )
		print command
		os.system(command)

		command = '../liblinear/predict \
../data/SVM_format_dev_w2v_rd%d.txt \
../data/head_user_SVM_format_random_train_w2v_rd%d_l%s_r%s_f%d.model result.txt |& tee \
result_head_user_w2v_rd%d_l%s_r%s_f%d_run%d.txt' %( 
	yelp_round,
	yelp_round, lambda_str, r_str , int(factor*10),
	yelp_round, lambda_str, r_str , int(factor*10), it )
		print command
		os.system(command)

	command = 'rm %s' %(random_head_swe)
	#print command
	os.system(command)

	command = 'rm ../data/head_user_SVM_format_random_train_swe_rd%d_l%s_r%s_f%d.model' %(yelp_round, lambda_str, r_str , int(factor*10))
	#print command
	os.system(command)

	acc_array = [0 for i in range(runs)]
	for it in range(runs):
		result_file = 'result_head_user_swe_rd%d_l%s_r%s_f%d_run%d.txt' %(yelp_round, lambda_str, r_str , int(factor*10), it)
		with open(result_file,"rb") as fin:
			for line in fin:
				line_array = line.split(' ')
				acc = line_array[-2]
				acc_array[it] = float(acc[:-1]) 
		command = 'rm %s' %(result_file)
		#print command
		os.system(command)

	# std = sqrt(mean(abs(x - x.mean())**2)).
	head_swe_std = np.std(acc_array)
	head_swe_mean = np.mean(acc_array)

	command = 'rm %s' %(random_head_w2v)
	#print command
	os.system(command)

	command = 'rm ../data/head_user_SVM_format_random_train_w2v_rd%d_l%s_r%s_f%d.model' %(yelp_round, lambda_str, r_str , int(factor*10))
	#print command
	os.system(command)

	acc_array = [0 for i in range(runs)]
	for it in range(runs):
		result_file = 'result_head_user_w2v_rd%d_l%s_r%s_f%d_run%d.txt' %( yelp_round, lambda_str, r_str ,int(factor*10), it )
		with open(result_file,"rb") as fin:
			for line in fin:
				line_array = line.split(' ')
				acc = line_array[-2]
				acc_array[it] = float(acc[:-1]) 
		command = 'rm %s' %(result_file)
		#print command
		os.system(command)

	# std = sqrt(mean(abs(x - x.mean())**2)).
	head_w2v_std = np.std(acc_array)
	head_w2v_mean = np.mean(acc_array)

	std_out = 'ht_std_out_rd%d_l%s_r%s_f%d.txt' %( yelp_round, lambda_str, r_str, int(factor*10))
	with open(std_out,'wb') as fo:
		fo.write(str( round(head_swe_std,4) )+'\n')
		fo.write(str( round(head_w2v_std,4) )+'\n')
		fo.write(str( round(tail_swe_std,4) )+'\n')
		fo.write(str( round(tail_w2v_std,4) )+'\n')

	mean_out = 'ht_mean_out_rd%d_l%s_r%s_f%d.txt' %( yelp_round, lambda_str, r_str, int(factor*10) )
	with open(mean_out,'wb') as fo:
		fo.write(str(head_swe_mean)+'\n')
		fo.write(str(head_w2v_mean)+'\n')
		fo.write(str(tail_swe_mean)+'\n')
		fo.write(str(tail_w2v_mean)+'\n')

	print 'Train on %d%% one-fifth data' %( int(factor*100) )
	print 'swe: head mean is %f' %(head_swe_mean)
	print 'swe: head std is %.4f' %(head_swe_std)

	print 'w2v: head mean is %f' %(head_w2v_mean)
	print 'w2v: head std is %.4f' %(head_w2v_std)

	print 'swe: tail mean is %f' %(tail_swe_mean)
	print 'swe: tail std is %.4f' %(tail_swe_std)

	print 'w2v: tail mean is %f' %(tail_w2v_mean)
	print 'w2v: tail std is %.4f' %(tail_w2v_std)

	command = 'rm result.txt'
	#print command
	os.system(command)

def print_results(yelp_round,lambda_str,r_str):
	factor_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	col, row = 10, 8;
	result_matrix = [['' for x in range(col)] for y in range(row)] 

	for i, factor in enumerate(factor_array):
		
		mean_out = 'ht_mean_out_rd%d_l%s_r%s_f%d.txt' %( yelp_round, lambda_str, r_str, int(factor*10) )
		with open(mean_out,'rb') as fin:
			for j, line in enumerate(fin):
				result_matrix[j][i]=line.strip('\n')
				
	
		std_out = 'ht_std_out_rd%d_l%s_r%s_f%d.txt' %( yelp_round, lambda_str, r_str, int(factor*10) )
		with open(std_out,'rb') as fin:
			for j, line in enumerate(fin):
				result_matrix[j+4][i]=line.strip('\n')
				
		command = 'rm %s' %(mean_out)
		os.system(command)

		command = 'rm %s' %(std_out)
		os.system(command)


	x = PrettyTable()
	
	x.add_column("Head",["SWE-mean", "SWE-std", "W2V-mean", "W2V-std"])

	for i in range(10):
		field = '%d%%' %((i+1)*10)
		x.add_column(field, [result_matrix[0][i],result_matrix[4][i],result_matrix[1][i],result_matrix[5][i]])
	
	print(x)

	table_txt = x.get_string()
	with open('head_tail_output.txt','a+') as fo:
		title = '\nrd%d_l%s_r%s\n' %(yelp_round, lambda_str, r_str)
		fo.write(title)
		fo.write(table_txt)

	x = PrettyTable()
	
	x.add_column("Tail",["SWE-mean", "SWE-std", "W2V-mean", "W2V-std"])

	for i in range(10):
		field = '%d%%' %((i+1)*10)
		x.add_column(field,[result_matrix[2][i],result_matrix[6][i],result_matrix[3][i],result_matrix[7][i]])

	print(x)
	
	table_txt = x.get_string()
	with open('head_tail_output.txt','a+') as fo:
		title = '\nrd%d_l%s_r%s\n' %(yelp_round, lambda_str, r_str)
		fo.write(title)
		fo.write(table_txt)

#-------------------------------------------------------------
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

	tune_para_SVM_swe(word_vec, user_vec, lambda_str, r_str, args.yelp_round)
	tune_para_SVM_w2v(word_vec_w2v, args.yelp_round)
	get_head_tail_review(args.yelp_round)
	SVM_format_w2v(args.yelp_round, word_vec_w2v)
	SVM_format_swe(args.yelp_round, word_vec, user_vec, lambda_str, r_str)
	
	for frac in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		run_SVM_frac(args.yelp_round, lambda_str, r_str, frac)

	print_results(args.yelp_round, lambda_str, r_str)
	

