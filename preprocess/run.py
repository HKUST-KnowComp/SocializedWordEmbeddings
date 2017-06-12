import os
##################################
# Processing data
##################################

command = 'cd ./preprocess'
print command
os.system(command)

command = 'python preprocess.py --input x --yelp_round 8'
print command
os.system(command)

command = 'python preprocess.py --input x --yelp_round 9'
print command
os.system(command)

'''
argument('--input', default=None, type=str,
                        help='Path to yelp dataset')
argument('--yelp_round', default=9, type=int, choices={8,9},
                        help="The round number of yelp data")
'''