import os

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))


# ファイルの場所

# ソース

PATH_TO_VGR_domain_text = _CUR_DIR + '/data/VGR_domain_text'

PATH_TO_VGR_domain_text2 = _CUR_DIR + '/data/VGR_domain_text2'

# 単語レベルと文字レベルの語彙

PATH_TO_WORD_VOCAB = _CUR_DIR + '/data/vocab/word_vocab.txt'

PATH_TO_CHAR_VOCAB = _CUR_DIR + '/data/vocab/char_vocab.txt'

# 入出力用データ

PATH_TO_X_TRAIN = _CUR_DIR + '/data/train/X_train.txt'

PATH_TO_Y_TRAIN = _CUR_DIR + '/data/train/y_train.txt'

PATH_TO_X_TEST = _CUR_DIR + '/data/test/X_test.txt'

PATH_TO_Y_TEST = _CUR_DIR + '/data/test/y_test.txt'