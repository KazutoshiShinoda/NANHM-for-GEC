import argparse
import MeCab
from gensim.corpora.dictionary import Dictionary
import config as cfg

char_rm_list = ["","\n"]

def _tokenize(text, tagger):
    sentence = []
    node = tagger.parse(text)
    node = node.split("\n")
    for i in range(len(node)):
        feature = node[i].split("\t")
        if feature[0] == "EOS":
            break
        sentence.append(feature[0])
    return sentence

def construct_vocab_and_train(tagger):
    f = open(cfg.PATH_TO_VGR_domain_text)
    g = open(cfg.PATH_TO_X_TRAIN, 'w')
    line = f.readline()
    word_dic = Dictionary()
    char_dic = Dictionary()
    word_dic.add_documents([["UNK","EOS"]])
    char_dic.add_documents([["UNK","BOW"]])
    while line:
        sentence = _tokenize(line, tagger)
        g.write(" ".join(sentence)+"\n")
        word_dic.add_documents([sentence])
        char_dic.add_documents([list(line)])
        line = f.readline()
    f.close
    g.close
    return list(word_dic.itervalues()), list(char_dic.itervalues())

def construct_test(tagger):
    f = open(cfg.PATH_TO_VGR_domain_text2)
    g = open(cfg.PATH_TO_X_TEST, 'w')
    line = f.readline()
    word_dic = Dictionary()
    char_dic = Dictionary()
    word_dic.add_documents([["UNK","EOS"]])
    char_dic.add_documents([["UNK","BOW"]])
    while line:
        sentence = _tokenize(line, tagger)
        g.write(" ".join(sentence)+"\n")
        word_dic.add_documents([sentence])
        char_dic.add_documents([list(line)])
        line = f.readline()
    f.close
    g.close
    return list(word_dic.itervalues()), list(char_dic.itervalues())

def main():
    parser = argparse.ArgumentParser(description='Construct vocabularies from corpus...')
    parser.add_argument('--dic-path', '-d', type=str, default=None,
                        help='dictionary path for MeCab')
    args = parser.parse_args()
    dic_path = args.dic_path
    if dic_path:
        tagger = MeCab.Tagger("-Ochasen -d {0}".format(dic_path))
    else:
        tagger = MeCab.Tagger("-Ochasen")

    word_dic, char_dic = construct_vocab_and_train(tagger)
    word_dic2, char_dic2 = construct_test(tagger)
    for ng in char_rm_list:
        if ng in char_dic:
            char_dic.remove(ng)
        if ng in char_dic2:
            char_dic2.remove(ng)
    
    f = open(cfg.PATH_TO_WORD_VOCAB, 'w')
    f.write("\n".join(word_dic))
    f.close
    g = open(cfg.PATH_TO_CHAR_VOCAB, 'w')
    g.write("\n".join(char_dic))
    g.close
    
    f = open(cfg.PATH_TO_WORD_VOCAB2, 'w')
    f.write("\n".join(word_dic2))
    f.close
    g = open(cfg.PATH_TO_CHAR_VOCAB2, 'w')
    g.write("\n".join(char_dic2))
    g.close
    
    print("Successfully constructed!")


if __name__ == '__main__':
    main()
