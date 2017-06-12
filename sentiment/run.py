import os

###########################################################
# Only use high users or only use tail users for training
# Corresponding to Figure 3 in the paper
# For re-production purpose, you may run the following parameters:
# Yelp 8
# --para_lambda 0.0  --para_r 0.25
# --para_lambda 1.0  --para_r 0.25
# Yelp 9
# --para_lambda 0.0  --para_r 0.25
# --para_lambda 1.0  --para_r 0.25
###########################################################

command = 'python head_tail.py --para_lambda 0.0 --para_r 0.25 --yelp_round 8'
print command
#os.system(command)

command = 'python head_tail.py --para_lambda 1.0 --para_r 0.25 --yelp_round 8'
print command
#os.system(command)

command = 'python head_tail.py --para_lambda 0.0 --para_r 0.25 --yelp_round 9'
print command
#os.system(command)

command = 'python head_tail.py --para_lambda 1.0 --para_r 0.25 --yelp_round 9'
print command
#os.system(command)


#########################################
# Sentiment Classification usign SVM
# Corresponding to Table 4 in the paper
# Yelp 8
# --para_lambda 0.0  --para_r 0.25
# --para_lambda 8.0  --para_r 0.25
# Yelp 9
# --para_lambda 0.0  --para_r 0.25
# --para_lambda 16.0  --para_r 4.0
#########################################

command = 'python sentiment.py --para_lambda 0.0 --para_r 0.25 --yelp_round 8'
print command
#os.system(command)

command = 'python sentiment.py --para_lambda 8.0 --para_r 0.25 --yelp_round 8'
print command
#os.system(command)

command = 'python sentiment.py --para_lambda 0.0 --para_r 0.25 --yelp_round 9'
print command
#os.system(command)

command = 'python sentiment.py --para_lambda 16.0 --para_r 4.0 --yelp_round 9'
print command
#os.system(command)


