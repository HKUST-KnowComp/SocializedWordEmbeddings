#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_USERS 687000 // Maximum number of users
#define MAX_USER_ID 200 // Maximum length of user ID string
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int user_hash_size = 3000000; // Maximum 3 * 0.7 = 2.1M users 
typedef float real; // Precision of float numbers

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

struct node {
    long long user_id;
    struct node *next;
};

char train_file[MAX_STRING], output_file[MAX_STRING],user_file[MAX_STRING],user_graph_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING], save_user_file[MAX_STRING], save_syn1_file[MAX_STRING];
struct vocab_word *vocab;
struct node** user_graph;
char** user;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
int *user_hash;
long long vocab_max_size = 1000, user_max_size = 1000, user_graph_max_size =1000, vocab_size = 0, user_size =0,  layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, lambda = 8, starting_alpha, sample = 1e-3, r = 0.25;
real *syn0, *syn1, *user0, *syn1neg, *expTable;
clock_t start;
int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void ReadWord(char *word, FILE *fin);
int GetWordHash(char *word);
int GetUserHash(char *word);
int SearchVocab(char *word);
int SearchUser(char *word);
int ReadWordIndex(FILE *fin);
int ReadUserID(FILE *fin);
int AddWordToVocab(char *word);
int AddIDToUser(char *word);
int VocabCompare(const void *a, const void *b);
void LearnVocabFromTrainFile();
void LearnUserFromUserFile();
void CreateBinaryTree();
void CreateAdjacentList();
void SortVocab();
void ReadVocab();
void SaveVocab();
void SaveUser();
void SaveContext();
void InitNet();
void *TrainModelThread(void *id);
void TrainModel();
int ArgPos(char *str, int argc, char **argv);


void InitUnigramTable() {
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns hash value of a user
int GetUserHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % user_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Returns position of a user in the user_vocabulary; if the user is not found, returns -1
int SearchUser(char *word) {
    unsigned int hash = GetUserHash(word);
    while(1) {
        if (user_hash[hash] == -1) return -1;
        if (!strcmp(word, user[user_hash[hash]])) return user_hash[hash]; //return the position of the user_list
        hash = (hash + 1) % user_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Reads a userID and returns its index in the user_vocabulary
int ReadUserID(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchUser(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word); // return the key by some hash function, but it may has conflict.
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size; // conflict
    vocab_hash[hash] = vocab_size - 1; //no conflict
    return vocab_size - 1;
}

// Adds a user id to the user_vocabulary
int AddIDToUser(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_USER_ID) length = MAX_USER_ID;
    user[user_size] = (char *)calloc(length, sizeof(char));
    strcpy(user[user_size], word);
    user_size++;
    // Reallocate memory if needed
    if (user_size + 2 >= user_max_size) {
        user_max_size += 1000;
        user = (char**)realloc(user, user_max_size * sizeof(char*));
        if (user == NULL){
            printf("Not enough memory for user");
        }
    }
    hash = GetUserHash(word); // return the key by some hash function, but it may has conflict.
    while (user_hash[hash] != -1) hash = (hash + 1) % user_hash_size; // conflict
    user_hash[hash] = user_size - 1; //no conflict
    return user_size - 1;
}

int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[2], vocab_size - 2, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    train_words = 0;
    for (a = 0; a < size; a++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[a].cn < min_count) && (a > 1)) {
            vocab_size--;
            free(vocab[a].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    for (a = 0; a < vocab_size; a++) {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int a, b = 2;
    unsigned int hash;
    for (a = 2; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
        vocab[b].cn = vocab[a].cn;
        vocab[b].word = vocab[a].word;
        b++;
    } else free(vocab[a].word);
    vocab_size = b;
    for (a = 2; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 2; a < vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;
        }
        vocab[a].codelen = i;
        vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab[a].code[i - b - 1] = code[b];
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

// Create an adjacent list to represent social graph using user_graph.txt which contains social information
void CreateAdjacentList(){
    char word[MAX_STRING];
    FILE *fin;
    int first;
    long long i,head;
    struct node *friend, *next_friend;
    user_graph = (struct node **)malloc(user_size*sizeof(struct node*));
    for(i=0;i<user_size;i++){
        user_graph[i]=NULL;
    }
    fin = fopen(user_graph_file, "rb");
    if (fin == NULL) {
        printf("ERROR: user graph file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_SET);
    while(1) {
        //read head
        ReadWord(word,fin);
        if (feof(fin)) break;
        head = SearchUser(word);
        // the head is not a user
        if (head == -1) {
            while(1){
                if(strcmp(word,"</s>")==0) break;
                ReadWord(word, fin);
                if (feof(fin)) break;
            }
            continue;
        }
        first =1;
        while (1){ // while word is not equal to </s>
            ReadWord(word, fin);
            if (feof(fin)) break;
            if (strcmp(word,"</s>")==0) break;
            i = SearchUser(word);
            if (i == -1) continue;
            next_friend = (struct node*)malloc(sizeof(struct node));
            next_friend->next =NULL;
            next_friend->user_id = i;
            if (first == 0){
                friend->next = next_friend;
                friend = next_friend;
            }else{
                user_graph[head] = next_friend;
                friend = next_friend;
                first = 0;
            }
        }
    }
    fclose(fin);
}

//create user_vocabulary from user_file.txt
void LearnUserFromUserFile() {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < user_hash_size; a++) user_hash[a] = -1;
    fin = fopen(user_file, "rb");
    if (fin == NULL) {
        printf("ERROR: user file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_SET);
    user_size =0; //global variable
    AddIDToUser((char *)"unknown_user_id");
    while (1) {
        ReadWord(word, fin);
        if (!strcmp(word,(char *)"</s>")) continue;
        if (feof(fin)) break;
        i = SearchUser(word);
        if (i == -1){
            AddIDToUser(word);  // increase user_size
        }
    }
    fclose(fin);
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *)"</s>");
    AddWordToVocab((char *)"unknown_word");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        if (SearchUser(word)!=-1) continue; 
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }
    vocab[SearchVocab((char *)"</s>")].cn = 0;
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
        printf("user size: %lld\n",user_size);
    }
    file_size = ftell(fin);
    fclose(fin);
}


void SaveVocab() {
  long long i, j;
  FILE *fo = fopen(save_vocab_file, "wb");
  fprintf(fo, "%lld\n", vocab_size);
  for (i = 0; i < vocab_size; i++) {
    fprintf(fo, "%s %lld %d ", vocab[i].word, vocab[i].cn ,(int)vocab[i].codelen);
    for (j = 0; j < (int)vocab[i].codelen; j++){
      fprintf(fo, "%d ", (int)vocab[i].point[j]);
    }
    for (j = 0; j < (int)vocab[i].codelen; j++){
      fprintf(fo, "%d ", (int)vocab[i].code[j]);
    }
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void SaveUser(){
    long long i,j;
    FILE *fo = fopen(save_user_file, "wb");
    i = SearchUser((char*)"unknown_user_id");
    if (i != -1) for(j = 0; j < layer1_size; j++) user0[i * layer1_size + j] = 0;
    fprintf(fo, "%lld %lld\n", user_size, layer1_size);
    for (i = 0; i < user_size; i++) {
        fprintf(fo, "%s ", user[i]);
        for (j =0; j < layer1_size; j++){
            fprintf(fo, "%lf ", user0[i * layer1_size + j]);
        }
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void SaveContext(){
    long long i,j;
    FILE *fo = fopen(save_syn1_file, "wb");
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    if (negative){
        for (i = 0; i < vocab_size; i++) {
            fprintf(fo, "%s ", vocab[i].word);
            for (j =0; j < layer1_size; j++) fprintf(fo, "%lf ", syn1neg[i * layer1_size + j]);
            fprintf(fo, "\n");
        }
    }
    else {
        for (i = 0; i < vocab_size; i++) {
            fprintf(fo, "%s ", vocab[i].word);
            for (j =0; j < layer1_size; j++) fprintf(fo, "%lf ", syn1[i * layer1_size + j]);
            fprintf(fo, "\n");
        }
    }
    fclose(fo);
}

void ReadVocab() {
    long long a, i = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word);
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fclose(fin);
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    fclose(fin);
}

void InitNet() {
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&user0, 128, (long long)user_size * layer1_size * sizeof(real));
    if (user0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    if (hs) {
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
            syn1[a * layer1_size + b] = 0;
    }
    if (negative>0) {
        a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
            syn1neg[a * layer1_size + b] = 0;
    }
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
    for (a = 0; a < user_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        user0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
    a = SearchUser((char*)"unknown_user_id");
    if (a != -1) for(b = 0; b < layer1_size; b++) user0[a * layer1_size + b] = 0;
    CreateBinaryTree();
    CreateAdjacentList();
}

void *TrainModelThread(void *id) {
    real l2norm, r_sq = r * r;
    real f, g;
    int  cut = 0, unknow = 0;
    // cut = 1 means the number of words in the review written by a user is greater than MAX_SENTENCE_LENGTH＋1, 
    // the review is cut into more than one sentence
    struct node* friend;
    long long next_start;
    long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0, user_id, unknow_id;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label, local_iter = iter;  
    unsigned long long next_random = (long long)id;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    real *user_temp = (real *)calloc(layer1_size, sizeof(real));
    unknow_id = SearchUser((char *)"unknown_user_id");
    next_start = file_size + 1;
    FILE *fi = fopen(train_file, "rb");
    if (fi == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    if ((int)id < num_threads -1 ){
        fseek(fi, file_size / (long long)num_threads * ((long long)id + 1), SEEK_SET);
        while((int)fgetc(fi) != (int)'\n'){
            if(feof(fi)) break;
        }
        next_start = ftell(fi);
    }

    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
    if ((int)id !=0){
        while((int)fgetc(fi) != (int)'\n'){
            if(feof(fi)) break;
        }
    }

    while (1) {
        if (word_count - last_word_count > 10000) { // update progress and alpha
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                       word_count_actual / (real)(iter * train_words + 1) * 100,
                       word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            } 
            alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        } 
        if (sentence_length == 0) { // read a new sentence
            if (cut == 0){
                user_id = ReadUserID(fi); 
                if (user_id != unknow_id){
                    unknow = 0;
                    for(c = 0; c < layer1_size; c++) user_temp[c] = 0;
                } 
                else
                    unknow = 1;     
            }
            else{
                cut = 0;
            }
            while (1) {
                word = ReadWordIndex(fi);
                if (feof(fi)) break;
                if (word == -1) continue;
                word_count++;
                if (word == 0) break;
                // The subsampling randomly discards frequent words while keeping the ranking same
                if (sample > 0) {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) { cut = 1; break;}
            }
            sentence_position = 0;
        } 
        if ( cut ==0 && ( feof(fi) || (long long)ftell(fi) > next_start ) ) { 
            // each thread process (train_words / number_threads) words, 
            // if it processed more than (train_words / number_threads) words, finish a iteration, go the beginning.
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
            if ((int)id !=0){
                while((int)fgetc(fi) != (int)'\n'){
                    if(feof(fi)) break;
                }
            }
            continue;
        }
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0; // sum of context words, only used by CBOW
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0; // gradient w.r.t X_w
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        
        
        if (cbow) {  //train the cbow architecture
            // in -> hidden in terms of word
            cw = 0;
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) { // calculate X_w
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                if (unknow == 1) for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size]; 
                else for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size] + user0[c + user_id * layer1_size]; //X_w
                cw++;
            }
            
            if (cw) { //cw is the number of context words of the target word
                for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
                if (hs) for (d = 0; d < vocab[word].codelen; d++) { // begine hireachical softmax
                    f = 0;
                    l2 = vocab[word].point[d] * layer1_size;
                    // Propagate hidden -> output :
                    for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
                    if (f <= -MAX_EXP) continue;
                    else if (f >= MAX_EXP) continue;
                    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];// f = sigmoid( X_w * theta )
                    // 'g' is the gradient multiplied by the learning rate  g = alpha* ( 1 - d - f )
                    g = (1 - vocab[word].code[d] - f) * alpha;
                    // Propagate errors output -> hidden
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2]; // e = e + g * theta
                    // Learn weights hidden -> output
                    for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c]; // theta = theta + g * X_w
                } // end of hireachical softmax
                // NEGATIVE SAMPLING
                if (negative > 0) for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * layer1_size;
                    f = 0;
                    for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else{ // g = alpha * ( 1 - d - f ), f = sigmoid ( X_w * theta )
                        g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                    }
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2]; // e = e + g * theta
                    for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];  // theta = theta + g * X_w
                }//end of negative sampling

                // hidden -> in
                for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                    c = sentence_position - window + a;
                    if (c < 0) continue;  // the number of previous words is less than the window size
                    if (c >= sentence_length) continue; // the number of after words is less than the window size
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c]/cw;
                }
                if (unknow != 1) for (c = 0; c < layer1_size; c++) user_temp[c] += neu1e[c];
            } // the end of cbow
        } else {  //train skip-gram
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                l1 = last_word * layer1_size; //l1 is the position of word;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                // HIERARCHICAL SOFTMAX
                if (hs) for (d = 0; d < vocab[word].codelen; d++) {
                    f = 0;
                    l2 = vocab[word].point[d] * layer1_size;
                    // Propagate hidden -> output
                    if (unknow == 1) for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
                    else for (c = 0; c < layer1_size; c++) f += ( syn0[c + l1]+ user0[user_id*layer1_size+c] ) * syn1[c + l2];
                    if (f <= -MAX_EXP) continue;
                    else if (f >= MAX_EXP) continue;
                    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]; // f = sigmoid( x * theta )
                    // 'g' is the gradient multiplied by the learning rate
                    g = (1 - vocab[word].code[d] - f) * alpha; // g = ( 1 - d - f ) * alpha
                    // Propagate errors output -> hidden
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2]; // e = e + g * theta
                    // Learn weights hidden -> output
                    if (unknow == 1) for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1] ;
                    else for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * (syn0[c + l1]+ user0[user_id*layer1_size+c]) ; // theta = theta + g * x
                }
                // NEGATIVE SAMPLING
                if (negative > 0) for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * layer1_size;
                    f = 0;
                    if (unknow == 1) for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                    else for (c = 0; c < layer1_size; c++) f += (syn0[c + l1]+ user0[user_id * layer1_size + c]) * syn1neg[c + l2];
                    
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha; // f = sigmoid ( theta * x )
                    
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2]; // e = e + g * theta
                    
                    if (unknow == 1) for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
                    else for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * ( syn0[c + l1] + user0[user_id*layer1_size+c] ); // theta = theta + g * x
                }
                // hidden -> input for each context word, update x
                for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
                if (unknow != 1) for (c = 0; c < layer1_size; c++) user_temp[c] += neu1e[c];
            }
        }// end of skip-gram
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            if (unknow != 1){
                if (cut == 0){
                    // update friend
                    friend = user_graph[user_id];
                    while (friend!=NULL){
                        long long friend_id = friend->user_id;
                        l2norm = 0;
                        for(c = 0; c < layer1_size; c++) {
                            user_temp[c] +=  -alpha * lambda * (user0[user_id*layer1_size+c]-user0[friend_id*layer1_size+c]);
                            user0[friend_id * layer1_size+c] += - alpha * lambda * (user0[friend_id*layer1_size+c] - user0[user_id*layer1_size+c]);
                            l2norm += user0[friend_id*layer1_size+c] * user0[friend_id*layer1_size+c];
                        }
                        if (l2norm > r_sq){
                            l2norm = sqrt(l2norm);
                            for (c = 0; c < layer1_size; c++) user0[friend_id*layer1_size+c] = user0[friend_id*layer1_size+c] * r / l2norm;
                        }
                        friend = friend -> next;
                    }
                    // update user 
                    l2norm = 0;
                    for(c = 0; c < layer1_size; c++) {
                        user0[user_id * layer1_size + c] += user_temp[c];
                        l2norm += user0[user_id * layer1_size + c] * user0[user_id * layer1_size + c];
                    }
                    if (l2norm > r_sq){
                        l2norm = sqrt(l2norm);
                        for (c = 0; c < layer1_size; c++) user0[user_id * layer1_size + c] = user0[user_id * layer1_size + c] * r / l2norm;    
                    }
                }
            }
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    free(user_temp);
    pthread_exit(NULL);
}

void TrainModel() {
    long long a, b, c, d;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;
    LearnUserFromUserFile();
    if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
    if (output_file[0] == 0) return;
    if (save_user_file[0] == 0) {
        printf("Do not specify a file to save user vector");
        return;
    }
    if (save_syn1_file[0] == 0){
        printf("Do not specify a file to save contextual vector");
        return;
    }
    InitNet();
    if (negative > 0) InitUnigramTable();
    start = clock();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    fo = fopen(output_file, "wb");
    if (classes == 0) {
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {
            fprintf(fo, "%s ", vocab[a].word);
            if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            fprintf(fo, "\n");
        }
    } else {
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;
        int *centcn = (int *)malloc(classes * sizeof(int));
        int *cl = (int *)calloc(vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
        for (a = 0; a < iter; a++) {
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
            for (b = 0; b < clcn; b++) centcn[b] = 1;
            for (c = 0; c < vocab_size; c++) {
                for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
            }
            for (b = 0; b < clcn; b++) {
                closev = 0;
                for (c = 0; c < layer1_size; c++) {
                    cent[layer1_size * b + c] /= centcn[b];
                    closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
                }
                closev = sqrt(closev);
                for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
            }
            for (c = 0; c < vocab_size; c++) {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++) {
                    x = 0;
                    for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                    if (x > closev) {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }
        // Save the K-means classes
        for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
    }
    if (save_user_file[0] != 0) SaveUser();
    if (save_syn1_file[0] != 0) SaveContext();
    if (save_vocab_file[0] != 0) SaveVocab();
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("Socialized Word Eembeddings\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");

        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-user <file>\n");
        printf("\t\tUse user id from <file> to build user_vocabulary\n");
        printf("\t-user-graph <file>\n");
        printf("\t\tUse social information from <file> to build social graph\n");
        
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-save-user <file>\n");
        printf("\t\tUse <file> to save the resulting user vectors\n");
        printf("\t-save-context <file>\n");
        printf("\t\tThe contextual vectors will be saved to <file>\n");

        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-lambda <float>\n");
        printf("\t\tSet the trade off parameter of regularization term; default is 8\n");
        printf("\t-r <float>\n");
        printf("\t\tSet the constraint of user's L2-norm; default is 0.25\n");
        
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");

        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");

        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        
        
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");


        printf("\nExamples:\n");
        printf("./swe -train data.txt -user user.txt -user-graph user_graph.txt -output vec.txt -save-user user_vec.txt ");
        printf("-save-context context_vec.txt -size 100 -window 5 -cbow 1 -hs 0 -negative 5 -lambda 8 -r 0.25 -threads 5 -iter 5\n\n");
        return 0;
    }
    output_file[0] = 0;
    user_graph_file[0] = 0;
    user_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    save_user_file[0] = 0;
    save_syn1_file[0] = 0;
    
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-user", argc, argv)) > 0) strcpy(user_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-user-graph", argc, argv)) > 0) strcpy(user_graph_file, argv[i + 1]);
    
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-user", argc, argv)) > 0) strcpy(save_user_file, argv[i + 1]);
    
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]); //dimension of word embeddings
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    
    if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    
    if (cbow) alpha = 0.05;
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-r", argc, argv)) > 0) r = atof(argv[i + 1]);
    
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);

    
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-context", argc, argv)) > 0) strcpy(save_syn1_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
    
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    user = (char**)calloc(user_max_size, sizeof(char*) );
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    user_hash = (int*)calloc(user_hash_size, sizeof(int));
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    TrainModel();
    return 0;
}


