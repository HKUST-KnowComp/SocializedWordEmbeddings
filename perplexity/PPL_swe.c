#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_EXP 6
#define EXP_TABLE_SIZE 1000
#define INFSMALL 1e-10
#define INFLARGE 1e10
#define MAX_STRING 300
#define MAX_SENTENCE_LENGTH 100
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int user_hash_size = 3000000;

struct vocab_word {
    long long cn;
    char *word;
};
struct vocab_word *vocab;
float *syn0, *syn1, *expTable, *user0;
long long vocab_size, layer1_size, user_size, vocab_count, user_count, sum = 0, infinity_cnt = 0;
char** user;
int *vocab_hash, *user_hash;
char syn0_file[MAX_STRING], syn1_file[MAX_STRING], test_file[MAX_STRING], user_file[MAX_STRING], output_file[MAX_STRING], sentence_file[MAX_STRING];

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

int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[ vocab_count ].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[ vocab_count ].word, word);
    vocab_count++;
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_count - 1;
    return vocab_count - 1;
}

int GetUserHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % user_hash_size;
    return hash;
}

int SearchUser(char *word) {
    unsigned int hash = GetUserHash(word);
    while(1) {
        if (user_hash[hash] == -1) return -1;
        if (!strcmp(word, user[user_hash[hash]])) return user_hash[hash]; //return the position of the user_list
        hash = (hash + 1) % user_hash_size;
    }
    return -1;
}

int AddIDToUser(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    user[user_count] = (char *)calloc(length, sizeof(char));
    strcpy(user[user_count], word);
    user_count++;
    hash = GetUserHash(word);
    while (user_hash[hash] != -1) hash = (hash + 1) % user_hash_size;
    user_hash[hash] = user_count - 1;
    return user_count - 1;
}

int Init(){
    long long a = 0, b = 0;
    char word[MAX_STRING];
    FILE* fin;
    
    //syn0
    fin = fopen(syn0_file, "rb");
    if (fin == NULL) {
        printf("ERROR: word vector file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_SET);
    ReadWord(word, fin);
    
    vocab_size = atoll( word );
    vocab = (struct vocab_word *)calloc(vocab_size, sizeof(struct vocab_word));
    
    ReadWord(word, fin);
    layer1_size = atoll( word );
    
    
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    
    
    while(!feof(fin)){
        ReadWord(word, fin);
        if (!strcmp(word, (char *)"</s>")){
            if (feof(fin)) break;
            ReadWord(word, fin);
            a = AddWordToVocab(word);
            b = 0;
            continue;
        }
        syn0 [ a * layer1_size + b ] = atof(word);
        b ++;
    }
    
    fclose(fin);
    
    //syn1
    vocab_count = 0;
    fin = fopen(syn1_file, "rb");
    if (fin == NULL) {
        printf("ERROR: contextual vector file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_SET);
    
    ReadWord(word, fin);
    if (atoll( word ) != vocab_size){
        printf("dimension conflict: syn1 vocab size is %lld global vocab_size is %lld\n", atoll(word), vocab_size);
        return -1;
    }
    ReadWord(word, fin);
    if (atoll( word ) != layer1_size){
        printf("dimension conflict: syn1 layer1 size is %lld, global layer1_size is %lld\n",atoll(word), vocab_size);
        return -1;
    }
    
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    
    while(!feof(fin)){
        ReadWord(word, fin);
        if (!strcmp(word, (char *)"</s>")){
            if (feof(fin)) break;
            ReadWord(word, fin);
            a = SearchVocab(word);
            b = 0;
            continue;
        }
        syn1 [ a * layer1_size + b ] = atof(word);
        b ++;
    }
    
    fclose(fin);
    
    //user
    fin = fopen(user_file, "rb");
    if (fin == NULL) {
        printf("ERROR: user vector file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_SET);
    
    ReadWord(word, fin);
    user_size = atoll( word );
    ReadWord(word, fin);
    
    a = posix_memalign((void **)&user0, 128, (long long)user_size * layer1_size * sizeof(float));
    if (user0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    user = (char**)calloc(user_size, sizeof(char*) );
    
    while(!feof(fin)){
        ReadWord(word, fin);
        if (!strcmp(word, (char *)"</s>")){
            if (feof(fin)) break;
            ReadWord(word, fin);
            a = AddIDToUser(word);
            b = 0;
            continue;
        }
        user0 [ a * layer1_size + b ] = atof(word);
        b ++;
    }
    
    fclose(fin);
    return 0;
}

long double n_gram_PPL_1(long long user_id, FILE* fin, int gram){ //P(first word | previous n words) ... P(</s> | previous n words)
    long long cur, next, a, c, word_count = 0, unknown_index;
    char word[MAX_STRING] = "start";
    long double prob = 1.0, normalization;
    float f = 0;
    float *neu0 = (float *)malloc(layer1_size * sizeof(float));
    long long *history = (long long *)malloc( MAX_SENTENCE_LENGTH * sizeof(long long));
    unknown_index = SearchVocab((char*)"unknown_word");
    int cw = 0;
    for(a = 0; a < layer1_size; a++) neu0[a] = 0;
    
    cur = SearchVocab((char*)"</s>");
    
    while( strcmp(word, (char *)"</s>") && !feof(fin)){ // not equal & not end of file
        ReadWord(word,fin);
        next = SearchVocab(word); // SearchVocab return the index of next
        if (next == -1) next = unknown_index;
        history[word_count] = cur; // keep all previous words
        word_count++; // how many words have been predicted
        for(a = 0; a < layer1_size; a++) neu0[a] += syn0[ cur * layer1_size + a];
        cw++;
        if (cw > gram) {
            for(a = 0; a < layer1_size; a++) neu0[a] -= syn0[ history[ word_count - gram - 1] * layer1_size + a];
            cw = gram;
        }
        normalization = 0;
        for (a = 0; a < vocab_size; a++){
            f = 0;
            for (c = 0; c < layer1_size; c++) f += (neu0[c] + cw * user0[ user_id * layer1_size + c ]) * syn1[a * layer1_size + c];
            normalization += exp(f/cw);
        }
        f = 0;
        for (c = 0; c < layer1_size; c++) f += (neu0[c] + cw * user0[ user_id * layer1_size + c ]) * syn1[next * layer1_size + c];
        if (normalization > 0) prob *= exp(f/cw) / normalization;
        else {
            printf("normalization <= 0\n");
            exit(1);
        }
        cur = next;
    }
    free(neu0);
    if (word_count > 0) {
        prob = powl(prob,-1.0/word_count);
        if (prob == INFINITY) {
            return -2;
        }
        else{
            return prob;
        }
    }
    else {
        return -1;
    }
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

float test(){
    FILE* fin, *fo = NULL, *fout;
    char word[MAX_STRING];
    fin = fopen(test_file, "rb");
    long long sentence_cnt = 0, user_id;
    long double val = 0, ppl = 0;
    if (output_file[0] != 0) fout = fopen(output_file,"wb");
    if (sentence_file[0] != 0) fo = fopen(sentence_file,"wb");
    if (fin == NULL) {
        printf("ERROR: test file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_SET);
    while(!feof(fin)){
        ReadWord(word, fin);
        user_id = SearchUser(word);
        if (user_id == -1) continue;
        sentence_cnt++;
        val = n_gram_PPL_1(user_id,fin,5);
        //printf("%.5lf\n", (double)val);
        if (sentence_file[0] != 0) {
            fprintf(fo, "%.5lf\n", (double)val);
            fflush(fo);
        }
        if(val == -2){
            infinity_cnt++;
            sentence_cnt--;
        }
        else if (val == -1) sentence_cnt--;
        else ppl += val;
    }
    
    if (sentence_file[0] != 0) fclose(fo);
    fclose(fin);
    if (output_file[0] != 0){ 
        fprintf(fout,"sentence_cnt is %lld\n",sentence_cnt);
        printf("sentence_cnt is %lld\n",sentence_cnt);
        fclose(fout);
    }
    if (sentence_cnt > 0) return ppl/sentence_cnt;
    else return -1;
}


int main(int argc, char **argv) {
    int i;
    long long a;
    long double ppl;
    FILE *fo;
    output_file[0] = 0;
    sentence_file[0] = 0;
    if ((i = ArgPos((char *)"-user", argc, argv)) > 0) strcpy(user_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-word-vec", argc, argv)) > 0) strcpy(syn0_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-test", argc, argv)) > 0) strcpy(test_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-context-vec", argc, argv)) > 0) strcpy(syn1_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-sentence", argc, argv)) > 0) strcpy(sentence_file, argv[i + 1]);
    user_hash = (int *)calloc(user_hash_size, sizeof(int));
    for (a = 0; a < user_hash_size; a++) user_hash[a] = -1;
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    Init();
    ppl = test();
    printf("Perplexity: %.5Lf\n",ppl);
    printf("%lld sentences has inifity perplexity!\n",infinity_cnt);
    if (output_file[0] != 0){
        fo = fopen(output_file,"a+");
        fprintf(fo,"-user %s -word-vec %s -test %s -context-vec %s\n", user_file, syn0_file, test_file, syn1_file);
        fprintf(fo,"Perplexity: %.5Lf\n",ppl);
        fprintf(fo,"%lld sentences has inifity perplexity!\n",infinity_cnt);
    }
    if (output_file[0] != 0) fclose(fo);
    return 0;
}
