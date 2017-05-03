# Socialized Word Embeddings
To complie swe.c

gcc swe.c -o swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

To train socialized word embeddings:


./swe -train data.txt -user user.txt -user-graph user_graph.txt -output vec.txt -save-user user_vec.txt -save-context context_vec.txt -size 100 -window 5 -cbow 1 -hs 0 -negative 5 -lambda 8 -r 0.25 -threads 5 -iter 5

