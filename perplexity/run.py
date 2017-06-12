import os
#########################################
# Perplexity 
# Corresponding to Table 2 in the paper
# For re-production purpose, you may run the following parameters:
# Yelp 8
# --para_lambda 0.0  --para_r 32
# --para_lambda 8.0  --para_r 32
# Yelp 9
# --para_lambda 0.0  --para_r 32.0
# --para_lambda 16.0  --para_r 16.0
#########################################

command =  'cd ../perplexity'
print command
os.system(command)

command = 'python ppl.py --para_lambda 0.0 --para_r 32.0 --yelp_round 8'
print command
os.system(command)

command = 'python ppl.py --para_lambda 8.0 --para_r 32.0 --yelp_round 8'
print command
os.system(command)

command = 'python ppl.py --para_lambda 0.0 --para_r 32.0 --yelp_round 9'
print command
os.system(command)

command = 'python ppl.py --para_lambda 16.0 --para_r 16.0 --yelp_round 9'
print command
os.system(command)

'''
argument('--para_lambda', default=None, type=float,
                        help='The trade off parameter between log-likelihood and regularization term')
argument('--para_r', default=None, type=float,
                        help='The constraint of L2-norm of the user vector')
argument('--yelp_round', default=9, type=int, choices={8,9},
                        help='The round number of yelp data')
'''
