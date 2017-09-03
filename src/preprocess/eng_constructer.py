import argparse
from gensim.corpora.dictionary import Dictionary
import config as cfg
import re

char_rm_list = ["","\n"]

def _tokenize(text):
    sentence = text.split()
    return sentence

def get_chars(text):
    text = re.sub(" ","",text)
    characters = list(text)
    return characters

def construct_vocab():
    f = open(cfg.PATH_TO_ENG_Y_TRAIN)
    g = open(cfg.PATH_TO_ENG_Y_TEST)
    word_dic = Dictionary()
    char_dic = Dictionary()
    word_dic.add_documents([["UNK","EOS"]])
    char_dic.add_documents([["UNK","BOW"]])
    
    line = f.readline()
    while line:
        sentence = _tokenize(line)
        word_dic.add_documents([sentence])
        char_dic.add_documents([get_chars(line)])
        line = f.readline()
    f.close
    
    line = g.readline()
    while line:
        sentence = _tokenize(line)
        word_dic.add_documents([sentence])
        char_dic.add_documents([get_chars(line)])
        line = g.readline()
    g.close
    return list(word_dic.itervalues()), list(char_dic.itervalues())

def main():
    word_dic, char_dic = construct_vocab()
    for ng in char_rm_list:
        if ng in char_dic:
            char_dic.remove(ng)
    
    f = open(cfg.PATH_TO_ENG_SOURCE_WORD_VOCAB, 'w')
    f.write("\n".join(word_dic))
    f.close
    g = open(cfg.PATH_TO_ENG_TARGET_WORD_VOCAB, 'w')
    g.write("\n".join(word_dic))
    g.close
    
    f = open(cfg.PATH_TO_ENG_SOURCE_CHAR_VOCAB, 'w')
    f.write("\n".join(char_dic))
    f.close
    g = open(cfg.PATH_TO_ENG_TARGET_CHAR_VOCAB, 'w')
    g.write("\n".join(char_dic))
    g.close
    
    print("Successfully constructed!")

if __name__ == '__main__':
    main()
