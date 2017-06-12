#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
char input_file[MAX_STRING], user_file[MAX_STRING], word_file[MAX_STRING];
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
void GetUserFile() {
    FILE *fin, *fo;
    char word[MAX_STRING];
    fo = fopen(user_file,"wb");
    fin = fopen(input_file,"rb");
    if (fin == NULL){
        printf("Input file is not found!");
        exit(-1);
    }
    fseek(fin, 0, SEEK_SET);

    ReadWord(word,fin);
    fprintf(fo, "%s\n", word);
    
    while(!feof(fin)){
        ReadWord(word,fin);
        if(!strcmp(word,(char*)"</s>")){
            ReadWord(word,fin);
            if (strcmp(word,(char *)"unknown_user_id")) fprintf(fo, "%s\n", word);
        }
    }
    fclose(fin);
    fclose(fo);
}

void GetW2VTrain() {
    FILE *fin, *fo;
    char word[MAX_STRING];
    fo = fopen(word_file,"wb");
    fin = fopen(input_file,"rb");
    if (fin == NULL){
        printf("Input file is not found!");
        exit(-1);
    }
    fseek(fin, 0, SEEK_SET);

    ReadWord(word,fin); //read a user_id
    
    while(!feof(fin)){
        ReadWord(word,fin);
        if(!strcmp(word,(char*)"</s>")){
            ReadWord(word,fin);
        }
        else fprintf(fo, "%s ", word);
    }
    fclose(fin);
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
        printf("Options:\n");
        printf("\t-input <file>\n");
        printf("\t\tUse text data from <file> to generate user_file and train_file\n");
        printf("\t-user <file>\n");
        printf("\t\tPath of user_file\n");
        printf("\t-word <file>\n");
        printf("\t\ttrain_file that can be userd for training word2vec\n");
    }
    user_file[0] = 0;
    word_file[0] = 0;
    if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-user", argc, argv)) > 0) strcpy(user_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-word", argc, argv)) > 0) strcpy(word_file, argv[i + 1]);
    if (user_file[0] != 0) GetUserFile();
    if (word_file[0] != 0) GetW2VTrain();
    return 0;
}











