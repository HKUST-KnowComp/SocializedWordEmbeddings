import sys, os
import argparse
import csv, math
from gensim.parsing import preprocessing
import csv
csv.field_size_limit(sys.maxsize)

def one_fifth_data(d_type, yelp_round):
	if d_type == 'train':
		input_file = '../data/SVM_train_rd%d.txt' %(yelp_round)
		output_file = '../data/SVM_train_one_fifth_rd%d.txt' %(yelp_round)
	elif d_type == 'dev':
		input_file = '../data/SVM_dev_rd%d.txt' %(yelp_round)
		output_file = '../data/SVM_dev_one_fifth_rd%d.txt' %(yelp_round)
	elif d_type == 'test':
		input_file = '../data/SVM_test_rd%d.txt' %(yelp_round)
		output_file = '../data/SVM_test_one_fifth_rd%d.txt' %(yelp_round)
	else:
		print 'No such data type %s' %(d_type)
		return None

	if os.path.isfile(output_file) == True:
		print 'The file %s already exists. If you want to generate it again, please delete it first.'  %(output_file)
		return None

	fin = open(input_file,"rb")
	fo = open(output_file,"wb")
	
	
	##################################
	# For the purpose of re-production 
	##################################
	for i, line in enumerate(fin):
		if i % 5 == 0:
			fo.write(line.strip('\n'))
			fo.write('\n')

	######################################################################
	# This is a fast way to get one-fifth data,
	# Although it cannot guarantee the exactly one-fifth proportion,
	# it can guarantee the randomness.
	'''
	for i, line in enumerate(fin):
		random_int = np.random.randint(5)
		if random_int == 0:
			fo.write(line.strip('\n'))
			fo.write('\n')
	fin.close()
	fo.close()
	'''
	######################################################################

def NN_preprocess(d_type, yelp_round):
	# preprocessing for sentiment classification using Deep Neural Network
	if d_type == 'train':
		input_file = 'train_rd%d.tmp' %(yelp_round)
		output_file = './NN_train_rd%d.tmp' %(yelp_round)
	elif d_type == 'dev':
		input_file = 'dev_rd%d.tmp' %(yelp_round)
		output_file = './NN_dev_rd%d.tmp' %(yelp_round)
	elif d_type == 'test':
		input_file = 'test_rd%d.tmp' %(yelp_round)
		output_file = './NN_test_rd%d.tmp' %(yelp_round)
	else:
		print 'No such dataset type: %s' %(d_type)
		return None
	command = 'java -jar Split_NN.jar %s %s' %(input_file,output_file)
	print command
	os.system(command)

	# remove stop words
	if d_type == 'train':
		input_file = './NN_train_rd%d.tmp' %(yelp_round)
		output_file = './NN_train_rd%d.txt' %(yelp_round)
	elif d_type == 'dev':
		input_file = './NN_dev_rd%d.tmp' %(yelp_round)
		output_file = './NN_dev_rd%d.txt' %(yelp_round)
	elif d_type == 'test':
		input_file = './NN_test_rd%d.tmp' %(yelp_round)
		output_file = './NN_test_rd%d.txt' %(yelp_round)
	else:
		print 'No such dataset type: %s' %(d_type)
		return None
	
	stop_file = 'english_stop.txt'
	
	fin = open(input_file,'rb')
	fs = open(stop_file,"rb")
	tar_file = open(output_file,'w+')

	with open(stop_file,"rb") as f:
		for i, l in enumerate(f):
			pass
		total = i + 1

	stop_word1 = ["" for i in range(total)]
	stop_word2 = ["" for i in range(total)]
	cnt1 = 0
	cnt2 = 0
	for l in fs:
		s= l.strip('\n')
		if "'" in s:
			stop_word1[cnt1] = s
			cnt1 =cnt1 + 1
		else:
			stop_word2[cnt2] = s
			cnt2 =cnt2+ 1

	user_flag = 0;
	review_flag = 0;
	start = 1
	begin_mark = str('@@@@@begin_mark@@@@@\n')
	for s in fin:
		if s == begin_mark:
			user_flag = 1
			continue
		if user_flag == 1:
			user_flag = 0
			if start != 1:
				tar_file.write('\n')
			else:
				start = 0
			user_star = s.strip('\n').split()
			if (len(user_star) < 2):
				print "there is no user_id & star rating following the start_mark!"
				print len(user_star)
				for i in range(len(user_star)):
					print user_star[i]
			tar_file.write(user_star[0]+'\t\t')
			tar_file.write(user_star[1]+'\t\t')
			continue
		try:
			s_array = s.encode('utf8').split()
			s=''
			if len(s_array) > 0:
				for ss in s_array:
					ss = ss.lower()
					if ss not in stop_word1:
						s=s+ss+' '
			else: 
				continue
			s = s.strip('\n')
			s = preprocessing.strip_punctuation(s)
			s = preprocessing.strip_non_alphanum(s)
			s = preprocessing.strip_numeric(s)
			s = preprocessing.strip_tags(s)
			s = preprocessing.strip_multiple_whitespaces(s)
			s_array = s.encode('utf8').split()
			s=''
			actual_word_cnt  = 0
			if len(s_array) > 0:
				for ss in s_array:
					if ss == "RRB" or ss =="LRB" or ss == "LCB" or ss == "RCB": # -LCB-, -LRB-, -RCB-, -RRB-
						continue
					if ss not in stop_word2:
						s=s+ss+' '
						actual_word_cnt = actual_word_cnt + 1
				if (actual_word_cnt > 0):
					tar_file.write(s[:-1])
					tar_file.write('#')
			else:
				continue
		except UnicodeDecodeError:
			continue
	fin.close()
	tar_file.close()

	command = 'rm %s' %(input_file)
	#print command
	os.system(command)

def Train_preprocess(yelp_round):
	
	input_file = 'train_rd%d.tmp' %(yelp_round)
	output_file = './swe_train_rd%d.txt' %(yelp_round)
	
	fin = open(input_file,'rb')
	fo = open(output_file,'wb')

	user_flag = 0;
	start = 1
	begin_mark = str('@@@@@begin_mark@@@@@\n')
	for s in fin:
		if s == begin_mark:
			user_flag = 1
			continue
		if user_flag == 1:
			user_flag = 0
			if start != 1:
				fo.write('\n')
			else:
				start = 0
			user_id = s.strip('\n').split()
			if len(user_id) < 1: print "there is no user_id following the start_mark!"
			fo.write(user_id[0]+' ')
			s = ''
			if len(user_id) <= 1:
				continue
			else:
				for i in range(len(user_id)-1):
					s = s + user_id[i+1] + ' '
		try:
			s = s.strip('\n')
			s = preprocessing.strip_punctuation(s)
			s = preprocessing.strip_non_alphanum(s)
			s = preprocessing.strip_numeric(s)
			s = preprocessing.strip_tags(s)
			s = preprocessing.strip_multiple_whitespaces(s)
			s_array = s.encode('utf8').split()	
		except UnicodeDecodeError:
			continue
		s=''
		actual_word_cnt  = 0
		for ss in s_array:
			ss = ss.lower()
			actual_word_cnt = actual_word_cnt + 1
			s=s+ss+' '
		if (actual_word_cnt>0):
			fo.write(s[:-1])
		else:
			continue

	fin.close()
	fo.close()

	# get user_file and train_file
	if os.path.isfile('./get_user_train_file') == False: 
		command = 'gcc get_user_file_w2v_train.c -o get_user_file_w2v_train -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
		print command
		os.system(command)
	
	user_file = 'user_file_rd%d.txt' %(yelp_round)
	w2v_train = './w2v_train_rd%d.txt' %(yelp_round)
	command = './get_user_file_w2v_train -input %s -user %s -word %s' %(output_file, user_file, w2v_train)
	print command
	os.system(command)

def SVM_preprocess(d_type, yelp_round):
	# preprocessing for sentiment classification using SVM
	# remove punctuation, tags, multiple spaces, tags, stop words, convert all words into lower case.
	if d_type == 'train':
		input_file = 'train_rd%d.tmp' %(yelp_round)
		output_file = './SVM_train_rd%d.txt' %(yelp_round)
	elif d_type == 'dev':
		input_file = 'dev_rd%d.tmp' %(yelp_round)
		output_file = './SVM_dev_rd%d.txt' %(yelp_round)
	elif d_type == 'test':
		input_file = 'test_rd%d.tmp' %(yelp_round)
		output_file = './SVM_test_rd%d.txt' %(yelp_round)
	else:
		print 'No such dataset type: %s' %(d_type)
		return None

	stop_file = 'english_stop.txt'

	with open(stop_file,"rb") as f:
		for i, l in enumerate(f):
			pass
		total = i + 1

	fin = open(input_file,"rb")
	fo = open(output_file,"wb")

	stop_word1 = ["" for i in range(total)]
	stop_word2 = ["" for i in range(total)]
	cnt1 = 0
	cnt2 = 0
	
	with open(stop_file,"rb") as fs:
		for l in fs:
			s= l.strip('\n')
			if "'" in s:
				stop_word1[cnt1] = s
				cnt1 =cnt1 + 1
			else:
				stop_word2[cnt2] = s
				cnt2 =cnt2+ 1

	user_flag = 0;
	start = 1
	begin_mark = str('@@@@@begin_mark@@@@@\n')
	for s in fin:
		if s == begin_mark:
			user_flag = 1
			continue
		if user_flag == 1:
			user_flag = 0
			if start != 1:
				fo.write('\n')
			else:
				start = 0
			user_id = s.strip('\n').split()
			if len(user_id) < 2: print "there is no user_id & star rating following the start_mark!"
			fo.write(user_id[0]+' '+user_id[1]+' ')
			s = ''
			if len(user_id) <= 2:
				continue
			else:
				for i in range(len(user_id)-2):
					s = s + user_id[i+2] + ' '
				#s = s[:-1]
		try:
			s_array = s.encode('utf8').split()
			s=''
			if len(s_array) > 0:
				for ss in s_array:
					ss = ss.lower()
					if ss not in stop_word1:
						s=s+ss+' '
			else: 
				continue
			s = s.strip('\n')
			if len(s) > 0:
				s = preprocessing.strip_punctuation(s)
				s = preprocessing.strip_non_alphanum(s)
				s = preprocessing.strip_numeric(s)
				s = preprocessing.strip_tags(s)
				s = preprocessing.strip_multiple_whitespaces(s)
				s_array = s.encode('utf8').split()
				s=''
				if len(s_array) > 0:
					for ss in s_array:
						if ss not in stop_word2:
							s=s+ss+' '
				else: 
					continue
			else: 
				continue
			if len(s) > 0:
				if s[-1] != ' ':
					s = s + ' '
			else:
				continue
			fo.write(s)
		except UnicodeDecodeError:
			continue

	fin.close()
	fo.close()

def PPL_preprocess(d_type, yelp_round):
	
	if d_type == 'dev':
		input_file = 'dev_rd%d.tmp' %(yelp_round)
		output_file = 'PPL_dev_rd%d.tmp' %(yelp_round)
	elif d_type == 'test':
		input_file = 'test_rd%d.tmp' %(yelp_round)
		output_file = 'PPL_test_rd%d.tmp' %(yelp_round)
	else:
		print 'No such dataset type: %s' %(d_type)
		return None
	
	command = 'java -jar Split_PPL.jar %s %s' %(input_file,output_file)
	print command
	os.system(command)

	if d_type == 'dev':
		input_file = 'PPL_dev_rd%d.tmp' %(yelp_round)
		output_file = 'PPL_dev_rd%d.tmp.tmp' %(yelp_round)
	elif d_type == 'test':
		input_file = 'PPL_test_rd%d.tmp' %(yelp_round)
		output_file = 'PPL_test_rd%d.tmp.tmp' %(yelp_round)
	else:
		print 'No such dataset type: %s' %(d_type)
		return None

	fin = open(input_file,'rb')
	fo = open(output_file,'wb')
	
	for s in fin:
		user_id = s.strip('\n').split()
		if len(user_id) <= 1: 
			print "there is no word or only user_id in this line!"
			continue
		else:
			fo.write(user_id[0]+' ')
			s = ''
			for i in range(len(user_id)-1):
				s = s + user_id[i+1]+' '
			s = s[:-1]
			try:
				s = preprocessing.strip_punctuation(s)
				s = preprocessing.strip_non_alphanum(s)
				s = preprocessing.strip_numeric(s)
				s = preprocessing.strip_tags(s)
				s = preprocessing.strip_multiple_whitespaces(s)
				s_array = s.encode('utf8').split()
			
			except UnicodeDecodeError:
				fo.write('\n')
				continue
		
			s=''
			actual_word_cnt  = 0
			if len(s_array) > 0:
				for ss in s_array:
					if ss == "RRB" or ss =="LRB" or ss == "LCB" or ss == "RCB":
						continue
					ss = ss.lower()
					s=s+ss+' '
					actual_word_cnt  = actual_word_cnt  + 1
				if actual_word_cnt > 0 :
					fo.write(s[:-1])
			fo.write('\n')
	
	fin.close()
	fo.close()

	command = 'rm %s' %(input_file)
	#print command
	os.system(command)

	# select a sentence for each user
	dic = {}
	lower_bound = 8
	upper_bound = 10

	if d_type == 'dev':
		input_file = './PPL_dev_rd%d.tmp.tmp' %(yelp_round)
		output_file = './PPL_dev_rd%d.txt' %(yelp_round)
	elif d_type == 'test':
		input_file = './PPL_test_rd%d.tmp.tmp' %(yelp_round)
		output_file = './PPL_test_rd%d.txt' %(yelp_round)

	fo = open(output_file,"wb")
	user_count = 0
	
	user_file = 'user_file_rd%d.txt' %(yelp_round)
	with open(user_file,"rb") as fin:
		for line in fin:
			user_id = line.strip('\n')
			if user_id not in dic.keys():
				dic[user_id] = user_count
				user_count = user_count + 1
		total = user_count
	print "total %d user" %(total)
	recorder = [0 for i in range(total)]
	
	with open(input_file,"rb") as fin:
		for i, line in enumerate(fin):
			array_line = line.strip('\n').split()
			if array_line[0] == "unknown_user_id":
				pass
			else:
				if recorder[ dic[ array_line[0] ] ] != 0:
					pass
				else:
					if(len(array_line) >= (lower_bound + 1) and len(array_line) <= (upper_bound + 1) ):
						fo.write(line.strip('\n'))
						fo.write('\n')
						recorder[ dic[ array_line[0] ] ]= 1
	
	go_on = 0
	count = 0
	for i in range(total):
		if recorder[i] == 0:
			go_on = 1
			count = count + 1
	
	if go_on == 1:
		with open(input_file,"rb") as fin:
			for i, line in enumerate(fin):
				array_line = line.strip('\n').split()
				if array_line[0] == "unknown_user_id":
					pass
				else:
					if recorder[ dic[ array_line[0] ] ] != 0:
						pass
					else:
						if(len(array_line) >= (lower_bound + 1 - 1) and len(array_line) <= (upper_bound + 1 + 1) ):
							fo.write(line.strip('\n'))
							fo.write('\n')
							recorder[ dic[ array_line[0] ] ]= 1
	
	
	go_on = 0
	count = 0
	for i in range(total):
		if recorder[i] == 0:
			go_on = 1
			count = count + 1
	
	if go_on == 1:
		with open(input_file,"rb") as fin:
			for i, line in enumerate(fin):
				array_line = line.strip('\n').split()
				if array_line[0] == "unknown_user_id":
					pass
				else:
					if recorder[ dic[ array_line[0] ] ] != 0:
						pass
					else:
						if(len(array_line) >= ( lower_bound + 1 - 2 ) and len(array_line) <= ( upper_bound + 1 + 2 ) ):
							fo.write(line.strip('\n'))
							fo.write('\n')
							recorder[ dic[ array_line[0] ] ]= 1			
	
	go_on = 0
	count = 0
	for i in range(total):
		if recorder[i] == 0:
			go_on = 1
			count = count + 1
	
	if go_on == 1:
		with open(input_file,"rb") as fin:
			for i, line in enumerate(fin):
				array_line = line.strip('\n').split()
				if array_line[0] == "unknown_user_id":
					pass
				else:
					if recorder[ dic[ array_line[0] ] ] != 0:
						pass
					else:
						if(len(array_line) >= ( lower_bound + 1 - 3 ) and len(array_line) <= ( upper_bound + 1 + 3 ) ):
							fo.write(line.strip('\n'))
							fo.write('\n')
							recorder[ dic[ array_line[0] ] ]= 1	
	
	go_on = 0
	count = 0
	for i in range(total):
		if recorder[i] == 0:
			go_on = 1
			count = count + 1
	
	if go_on == 1:
		with open(input_file,"rb") as fin:
			for i, line in enumerate(fin):
				array_line = line.strip('\n').split()
				if array_line[0] == "unknown_user_id":
					pass
				else:
					if recorder[ dic[ array_line[0] ] ] != 0:
						pass
					else:
						fo.write(line.strip('\n'))
						fo.write('\n')
						recorder[ dic[ array_line[0] ] ]= 1	
	
	go_on = 0
	count = 0
	for i in range(total):
		if recorder[i] == 0:
			go_on = 1
			count = count + 1
	if go_on == 1:
		print "ERROR"
	fo.close()
	
	command = 'rm %s' %(input_file)
	#print command
	os.system(command)

def split(input_path, yelp_round):
	# split the data to be 8:1:1 for training, developing, and testing.
	dic={}
	thre = 30 # those people who published reveiws less than 30 will be treated as unknown user, we do not learn user vector for them.
	train_path = 'train_rd%d.tmp' %(yelp_round)

	dev_path = 'dev_rd%d.tmp' %(yelp_round)

	test_path = 'test_rd%d.tmp' %(yelp_round)

	train_fo = open(train_path, "wb")
	dev_fo = open(dev_path, "wb")
	test_fo = open(test_path, "wb")
	
	if yelp_round == 8:
		
		known_recorder = [0 for i in range(687000)]
		count_review = [0 for i in range(687000)]
		star_list = [0 for i in range(687000)]
		unknown_user_review_cnt = 0;
		unknown_recorder = 0;
		
		#count_review
		with open( os.path.join(input_path,'yelp_academic_dataset_review.csv') ) as f:
			f_csv = csv.reader(f)
			count = -1;
			for row in f_csv:
				if count==-1:
					count=count+1
					continue
				
				if row[0] in dic.keys():
					count_review[ dic[ row[0] ] ] = count_review[ dic[ row[0] ] ] + 1;
				else:
					if count%10000==0:
						print (str(count) + '/687000')
					dic[row[0]]=count
					count_review[ dic[ row[0] ] ] = 1
					count = count + 1
		
		for i in range(687000):
			if (count_review[i]<thre):
				unknown_user_review_cnt = unknown_user_review_cnt + count_review[i]
		
		with open( os.path.join(input_path,'yelp_academic_dataset_review.csv') ) as f:
			f_csv = csv.reader(f)
			count = -1;
			for row in f_csv:
				if count == -1:
					count = count + 1
					continue
				if row[0]=='user_id':
					print 'error: include user_id at',count
				if count_review[ dic[ row[0] ] ] < thre:
					if unknown_recorder < math.floor(unknown_user_review_cnt * 0.1):
						test_fo.write("@@@@@begin_mark@@@@@\n")
						test_fo.write("unknown_user_id ")
						test_fo.write(row[6]+" ")
						test_fo.write(row[2])
						test_fo.write('\n')
					elif unknown_recorder < math.floor(unknown_user_review_cnt * 0.2):
						dev_fo.write("@@@@@begin_mark@@@@@\n")
						dev_fo.write("unknown_user_id ")
						dev_fo.write(row[6]+" ")
						dev_fo.write(row[2])
						dev_fo.write('\n')
					else:
						train_fo.write("@@@@@begin_mark@@@@@\n")
						train_fo.write("unknown_user_id ")
						train_fo.write(row[6]+" ")
						train_fo.write(row[2])
						train_fo.write('\n')
					unknown_recorder = unknown_recorder + 1
				else:
					if known_recorder[ dic[ row[0] ] ] < math.floor(count_review[ dic[ row[0] ] ] * 0.1):
						test_fo.write("@@@@@begin_mark@@@@@\n")
						test_fo.write(row[0]+" ")
						test_fo.write(row[6]+" ")
						test_fo.write(row[2])
						test_fo.write('\n')
					elif known_recorder[ dic[ row[0] ] ] < math.floor(count_review[ dic[ row[0] ] ] * 0.2):
						dev_fo.write("@@@@@begin_mark@@@@@\n")
						dev_fo.write(row[0]+" ")
						dev_fo.write(row[6]+" ")
						dev_fo.write(row[2])
						dev_fo.write('\n')
					else:
						train_fo.write("@@@@@begin_mark@@@@@\n")
						train_fo.write(row[0]+" ")
						train_fo.write(row[6]+" ")
						train_fo.write(row[2])
						train_fo.write('\n')
					known_recorder[ dic[ row[0] ] ] = known_recorder[ dic[ row[0] ] ] + 1

		
	else:
		known_recorder = [0 for i in range(1029432)]
		count_review = [0 for i in range(1029432)]
		star_list = [0 for i in range(1029432)]
		unknown_user_review_cnt = 0;
		unknown_recorder = 0;
		
		#count_review
		with open( os.path.join( input_path,'yelp_academic_dataset_review.csv') ) as f:
			f_csv = csv.reader(f)
			count = -1;
			for row in f_csv:
				if count==-1:
					count=count+1
					continue
				if row[1] in dic.keys():
					count_review[ dic[ row[1] ] ] = count_review[ dic[ row[1] ] ] + 1;
				else:
					if count%10000==0:
						print ( str(count) +'/1029432' )
					dic[row[1]]=count
					count_review[ dic[ row[1] ] ] = 1
					count = count + 1
		
		for i in range(1029432):
			if (count_review[i]<thre):
				unknown_user_review_cnt = unknown_user_review_cnt + count_review[i]
		
		with open(os.path.join( input_path,'yelp_academic_dataset_review.csv')) as f:
			f_csv = csv.reader(f)
			count = -1;
			for row in f_csv:
				if count == -1:
					count = count + 1
					continue
				if row[1]=='user_id':
					print 'error: include user_id at',count
				if count_review[ dic[ row[1] ] ] < thre:
					if unknown_recorder < math.floor(unknown_user_review_cnt * 0.1):
						test_fo.write("@@@@@begin_mark@@@@@\n")
						test_fo.write("unknown_user_id ")
						test_fo.write(row[5]+" ")
						test_fo.write(row[3])
						test_fo.write('\n')
					elif unknown_recorder < math.floor(unknown_user_review_cnt * 0.2):
						dev_fo.write("@@@@@begin_mark@@@@@\n")
						dev_fo.write("unknown_user_id ")
						dev_fo.write(row[5]+" ")
						dev_fo.write(row[3])
						dev_fo.write('\n')
					else:
						train_fo.write("@@@@@begin_mark@@@@@\n")
						train_fo.write("unknown_user_id ")
						train_fo.write(row[5]+" ")
						train_fo.write(row[3])
						train_fo.write('\n')
					unknown_recorder = unknown_recorder + 1
				else:
					if known_recorder[ dic[ row[1] ] ] < math.floor(count_review[ dic[ row[1] ] ] * 0.1):
						test_fo.write("@@@@@begin_mark@@@@@\n")
						test_fo.write(row[1]+" ")
						test_fo.write(row[5]+" ")
						test_fo.write(row[3])
						test_fo.write('\n')
					elif known_recorder[ dic[ row[1] ] ] < math.floor(count_review[ dic[ row[1] ] ] * 0.2):
						dev_fo.write("@@@@@begin_mark@@@@@\n")
						dev_fo.write(row[1]+" ")
						dev_fo.write(row[5]+" ")
						dev_fo.write(row[3])
						dev_fo.write('\n')
					else:
						train_fo.write("@@@@@begin_mark@@@@@\n")
						train_fo.write(row[1]+" ")
						train_fo.write(row[5]+" ")
						train_fo.write(row[3])
						train_fo.write('\n')
					known_recorder[ dic[ row[1] ] ] = known_recorder[ dic[ row[1] ] ] + 1

	test_fo.close()
	dev_fo.close()
	train_fo.close()

def user_graph(input_dir, yelp_round):
	output_file = 'user_graph_rd%d.txt' %(yelp_round) 
	user_csv_file = os.path.join(input_dir, 'yelp_academic_dataset_user.csv')
	
	if yelp_round == 8:
		fo = open(output_file, 'wb')
		with open(user_csv_file) as f:
			f_csv = csv.reader(f)
			headers = next(f_csv)
			for row in f_csv:
				fo.write(row[16] +' ');
				friends = row[3].split(', ');
				num_friends = len(friends);
				for i in range(0,num_friends):
					friends[i]=friends[i].strip('\'');
					friends[i]=friends[i].strip('[\'');
					friends[i]=friends[i].strip('\']');
					fo.write(friends[i]+' ');
				fo.write('\n');
				fo.flush()
		fo.close()
	elif yelp_round == 9:
		fo = open(output_file, 'wb')
		with open(user_csv_file) as f:
			f_csv = csv.reader(f)
			headers = next(f_csv)	
			for row in f_csv:
				fo.write(row[15] +' ');
				friends = row[17].split(', ');
				num_friends = len(friends);
				for i in range(0,num_friends):
					friends[i]=friends[i].strip('\'');
					friends[i]=friends[i].strip('[\'');
					friends[i]=friends[i].strip('\']');
					fo.write(friends[i]+' ');
				fo.write('\n');
				fo.flush()
		fo.close()
	else:
		print 'No such round in yelp: %s' %(yelp_round)
		return None

def clean(yelp_round):
	# remove tmp file
	train_path = 'train_rd%d.tmp' %(yelp_round)

	dev_path = 'dev_rd%d.tmp' %(yelp_round)

	test_path = 'test_rd%d.tmp' %(yelp_round)

	command = 'rm %s' %(train_path)
	#print command
	#os.system(command)

	command = 'rm %s' %(dev_path)
	#print command
	#os.system(command)

	command = 'rm %s' %(test_path)
	#print command
	#os.system(command)

def move_data(yelp_round):
	if yelp_round == 8:
		files = ['SVM_train_rd8.txt','SVM_dev_rd8.txt','SVM_test_rd8.txt',
	'PPL_dev_rd8.txt','PPL_test_rd8.txt',
	'NN_train_rd8.txt','NN_dev_rd8.txt','NN_test_rd8.txt',
	'user_file_rd8.txt','user_graph_rd8.txt',
	'swe_train_rd8.txt','w2v_train_rd8.txt']
	elif yelp_round == 9:
		files = ['SVM_train_rd9.txt','SVM_dev_rd9.txt','SVM_test_rd9.txt',
	'PPL_dev_rd9.txt','PPL_test_rd9.txt',
	'NN_train_rd9.txt','NN_dev_rd9.txt','NN_test_rd9.txt',
	'user_file_rd9.txt','user_graph_rd9.txt',
	'swe_train_rd9.txt','w2v_train_rd9.txt']
	else:
		print 'No round number %d' %(yelp_round)
		return None

	for f in files:
		command = 'mv %s ../data' %(f)
		os.system(command)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Preprocess Yelp Data')

	parser.add_argument('--input', default=None, type=str,
                        help='Path to yelp dataset')

	parser.add_argument('--yelp_round', default=9, type=int, choices={8,9},
                        help="The round number of yelp data")


	args = parser.parse_args()

	parser.print_help()

	
	split(args.input, args.yelp_round)

	user_graph(args.input, args.yelp_round)

	SVM_preprocess('train',args.yelp_round)
	SVM_preprocess('dev',args.yelp_round)
	SVM_preprocess('test',args.yelp_round)

	Train_preprocess(args.yelp_round)

	PPL_preprocess('dev',args.yelp_round)
	PPL_preprocess('test',args.yelp_round)
	
	NN_preprocess('train',args.yelp_round)
	NN_preprocess('dev',args.yelp_round)
	NN_preprocess('test',args.yelp_round)

	
	clean(args.yelp_round)
	
	move_data(args.yelp_round)
	
	one_fifth_data('train',args.yelp_round)
	one_fifth_data('dev',args.yelp_round)
	one_fifth_data('test',args.yelp_round)
	



