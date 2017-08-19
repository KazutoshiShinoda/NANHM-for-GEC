import argparse
import MeCab
from gensim.corpora.dictionary import Dictionary
import config as cfg



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

def construct(tagger):
    f = open(cfg.PATH_TO_VGR_domain_text)
    line = f.readline()
    dic = Dictionary()
    i = 0
    while line:
        sentence = _tokenize(line, tagger)
        dic.add_documents([sentence])
        line = f.readline()
        i += 1
        if i == 100:
            break
    f.close
    return list(dic.itervalues())

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

    dict = construct(tagger)

    f = open(cfg.PATH_TO_VOCAB, 'w')
    f.write("\n".join(dict))
    f.close
    print("Successfully constructed!")


if __name__ == '__main__':
    main()
