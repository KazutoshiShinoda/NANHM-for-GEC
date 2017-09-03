#!/usr/bin/env python

import argparse

from nltk.translate import bleu_score
import numpy as np
import progressbar
import six
import config as cfg
from datetime import datetime as dt

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from my_chainer.links.connection import n_step_gru as My
from chainer import reporter
from chainer import training
from chainer.training import extensions
from chainer import serializers


#word-level
UNK = 0
EOS = 1

#char-level
UNK = 0
BOW = 1   # Boundary 0f Words

#Hyper parameter
Alpha = 0.5
Beta = 0.5

target_words = {}
target_word_ids = {}
target_chars = {}
target_char_ids = {}

char_hidden = []

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs

def convert_unk(embed, cs):
    cs = F.broadcast(cs)
    cexs = embed(cs)
    return (cexs,)

def get_unk_hidden_vector(ex, pos, x, embed, encoder, char_hidden):
    if len(pos)==0:
        char_hidden.extend([0]*len(ex))
        return ex
    else:
        hidden=[]
        for j in range(len(ex)):
            if j in pos:
                x_ = x[pos==j]
                cexs = convert_unk(embed, x_[0])
                hx, os = encoder(None, cexs)
                hidden.append(os)
                ex[j].data = hx[-1][-1].data
            else:
                hidden.append(0)
        char_hidden.extend(hidden)
        return ex

class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_source_char, n_target_char, n_units):
        super(Seq2seq, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units),
            embed_y=L.EmbedID(n_target_vocab, n_units * 2),
            embed_xc=L.EmbedID(n_source_char, n_units),
            embed_yc=L.EmbedID(n_target_char, n_units),
            encoder_f=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            encoder_b=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            char_encoder=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            decoder=My.NStepGRU(n_layers, n_units * 2, n_units * 2, 0.1),
            char_decoder=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            char_att_decoder=My.NStepGRU(n_layers, n_units, n_units, 0.1),
            W=L.Linear(n_units * 2, n_target_vocab),
            W_hat=L.Linear(n_units * 4, n_units),
            W_char=L.Linear(n_units, n_target_char),
        )
        self.n_layers = n_layers
        self.n_units = n_units
        
    def __call__(self, xs, ys):
        char_hidden=[]
        
        wxs = [x[0] for x in xs]
        unk_xs = [x[1] for x in xs]
        unk_pos = [np.where(x==UNK)[0] for x in wxs]
        concat_wxs = np.concatenate(wxs)
        
        wys = [y[0] for y in ys]
        unk_ys = [y[1] for y in ys]
        
        eos = self.xp.array([EOS], 'i')
        ys_in = [F.concat([eos, y[0]], axis=0) for y in ys]
        ys_out = [F.concat([y[0], eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, wxs)
        # Convert an UNK word vector into a char-hidden vector
        exs = list(map(lambda s, t, u: get_unk_hidden_vector(s, t, u, self.embed_xc, self.char_encoder, char_hidden) , exs, unk_pos, unk_xs))
        exs_f = exs
        exs_b = [ex[::-1] for ex in exs]
        eys = sequence_embed(self.embed_y, ys_in)
        
        batch = len(xs)
        # None represents a zero vector in an encoder.
        _, hf = self.encoder_f(None, exs_f)
        _, hb = self.encoder_b(None, exs_b)
        # 隠れ状態ベクトルの集合
        ht = list(map(lambda x,y: F.concat([x, y], axis=1), hf, hb))
        
        h_list, h_bar_list, c_s_list, z_s_list = self.decoder(None, ht, eys)
        
        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        os = h_list
        os_len = [len(s) for s in os]
        os_section = np.cumsum(os_len[:-1])
        concat_os = F.concat(os, axis=0)
        concat_os_out = self.W(concat_os)
        concat_ys_out = F.concat(ys_out, axis=0)
        
        loss_w = 0
        loss_c1 = 0
        loss_c2 = 0
        
        # If predicted word is UNK
        concat_pred_w = F.argmax(concat_os_out, axis=1)
        #concat_isUNK = concat_pred_w==0
        
        is_unk = concat_pred_w.data==UNK
        if UNK in concat_pred_w.data:
            print(True)
            ##案２：
            #全てconcat
            #総単語数*2048
            concat_c_s = F.concat(c_s_list, axis=0)
            concat_h_bar = F.concat(h_bar_list, axis=0)
            
            c_ss = concat_c_s[is_unk]
            h_bars = concat_h_bar[is_unk]
            c = F.concat([c_ss, h_bars], axis=1)
            ds_hats=F.relu(self.W_hat(c))
            
            z_s_len = [len(z_s) - 1 for z_s in z_s_list]
            z_s_section = np.cumsum(z_s_len[:-1])
            valid_z_s_section = np.insert(z_s_section, 0, 0)
            abs_z_s_list = [z_s_list[i] + valid_z_s_section[i] for i in range(len(z_s_list))]
            concat_z_s = F.concat(abs_z_s_list, axis=0)
            z_ss = concat_z_s[is_unk]
            
            true_wys = concat_ys_out[is_unk]
            #"予想単語==UNK"の各ケースについて個別に処理
            for i,wy in enumerate(true_wys):
                bow = self.xp.array([BOW], 'i')
                wy = int(wy.data)
                print(target_words[wy])
                if wy != UNK and wy != EOS:
                    cys = np.array([[target_char_ids[c] for c in list(target_words[wy])]], np.int32)
                elif wy == UNK:
                    #本来ありえない
                    cys = np.array([[target_char_ids['UNK']]], np.int32)
                elif wy == EOS:
                    cys = np.array([[target_char_ids['BOW']]], np.int32)
                cys_in = [F.concat([bow, y], axis=0) for y in cys]
                cys_out = [F.concat([y, bow], axis=0) for y in cys]
                concat_cys_out = F.concat(cys_out, axis=0)
                ceys = sequence_embed(self.embed_yc, cys_in)
                z_s = int(z_ss[i].data)
                
                ds_hat =F.reshape(ds_hats[i], (1, 1, ds_hats[i].shape[0]))
                if concat_wxs[z_s] != UNK:
                    #attentionなし文字ベースdecoder
                    _, cos = self.char_decoder(ds_hat, ceys)
                    print("attなし")
                    concat_cos = F.concat(cos, axis=0)
                    concat_cos_out=self.W_char(concat_cos)
                    loss_c1= loss_c1 + F.sum(F.softmax_cross_entropy(
                        concat_cos_out, concat_cys_out, reduce='no'))
                else:
                    #attentionあり文字ベースdecoder
                    ht = char_hidden[z_s]
                    h_list, h_bar_list, c_s_list, z_s_list = self.char_att_decoder(ds_hat, ht, ceys)
                    print("attあり")
                    concat_cos = F.concat(h_list, axis=0)
                    concat_cos_out=self.W_char(concat_cos)
                    loss_c2 = loss_c2 + F.sum(F.softmax_cross_entropy(
                        concat_cos_out, concat_cys_out, reduce='no'))
        else:
            print(False)
        
        loss_w = F.sum(F.softmax_cross_entropy(
            concat_os_out[is_unk!=1], concat_ys_out[is_unk!=1], reduce='no'))
        
        loss = F.sum(loss_w + Alpha * loss_c1 + Beta * loss_c2) / batch

        reporter.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        reporter.report({'perp': perp}, self)
        print("loss",loss)
        print()
        return loss

    def translate(self, xs, max_length=100):
        print("Now translating")
        batch = len(xs)
        print("batch",batch)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            char_hidden=[]
            
            wxs = [x[0] for x in xs]
            unk_xs = [x[1] for x in xs]
            unk_pos = [np.where(x==UNK)[0] for x in wxs]
            wx_len = [len(wx) for wx in wxs]
            wx_section = np.cumsum(wx_len[:-1])
            valid_wx_section = np.insert(wx_section, 0, 0)
            concat_wxs = np.concatenate(wxs)
            
            exs = sequence_embed(self.embed_x, wxs)
            exs = list(map(lambda s, t, u: get_unk_hidden_vector(s, t, u, self.embed_xc, self.char_encoder, char_hidden) , exs, unk_pos, unk_xs))
            
            exs_f = exs
            exs_b = [ex[::-1] for ex in exs]
            _, hf = self.encoder_f(None, exs_f)
            _, hb = self.encoder_b(None, exs_b)
            ht = list(map(lambda x,y: F.concat([x, y], axis=1), hf, hb))
            ys = self.xp.full(batch, EOS, 'i')
            result = []
            h_list = None
            for a in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                if h_list==None:
                    h0 = h_list
                else:
                    h0 = F.transpose_sequence(h_list)[-1]
                    h0 = F.reshape(h0, (self.n_layers, h0.shape[0], h0.shape[1]))
                #h0 : {type:variable, shape:(n_layers*batch*dimentionality)} or None
                h_list, h_bar_list, c_s_list, z_s_list = self.decoder(h0, ht, eys)
                
                os = h_list
                concat_os = F.concat(os, axis=0)
                concat_os_out = self.W(concat_os)
                concat_pred_w = self.xp.argmax(concat_os_out.data, axis=1).astype('i')
                is_unk = concat_pred_w==UNK
                
                if UNK in concat_pred_w:
                    N = np.sum(is_unk)
                    
                    concat_c_s = F.concat(c_s_list, axis=0)
                    concat_h_bar = F.concat(h_bar_list, axis=0)

                    c_ss = concat_c_s[is_unk]
                    h_bars = concat_h_bar[is_unk]
                    c = F.concat([c_ss, h_bars], axis=1)
                    ds_hats=F.relu(self.W_hat(c))
                    
                    abs_z_s_list = [z_s_list[i] + valid_wx_section[i] for i in range(len(z_s_list))]
                    concat_z_s = F.concat(abs_z_s_list, axis=0)
                    z_ss = concat_z_s[is_unk]
                    
                    #各UNK単語について
                    results_c = []
                    for i in range(N):
                        result_c = []
                        cy = self.xp.full(1, BOW, 'i')
                        cy = F.split_axis(cy, 1, 0)
                        cey = sequence_embed(self.embed_yc, cy)
                        z_s = int(z_ss[i].data)
                        ds_hat = F.reshape(ds_hats[i], (1, 1, ds_hats[i].shape[0]))
                        
                        if concat_wxs[z_s] != UNK:
                            for b in range(10):
                                #attentionなし文字ベースdecoder
                                ds_hat, cos = self.char_decoder(ds_hat, cey)
                                cos_out = self.W_char(cos[0])
                                pred_cos = self.xp.argmax(cos_out.data, axis=1).astype('i')
                                cey = self.embed_yc(pred_cos)
                                print(pred_cos)
                                print(target_chars[pred_cos])
                                result_c.append(pred_cos)
                        else:
                            c_ht = char_hidden[z_s]
                            for b in range(10):
                                #attentionあり文字ベースdecoder
                                if b==0:
                                    c_h0 = ds_hat
                                else:
                                    c_h0 = F.transpose_sequence(h_list)[-1]
                                    c_h0 = F.reshape(c_h0, (self.n_layers, c_h0.shape[0], c_h0.shape[1]))
                                c_h_list, c_h_bar_list, c_c_s_list, c_z_s_list = self.char_att_decoder(c_h0, c_ht, cey)
                                cos_out = self.W_char(h_list[-1])
                                pred_cos = self.xp.argmax(cos_out.data, axis=1).astype('i')
                                cey = self.embed_yc(pred_cos)
                                print(pred_cos)
                                print(target_chars[pred_cos])
                                result_c.append(pred_cos)
                        r = ""
                        for c in result_c:
                            if c == BOW:
                                break
                            r+=target_chars.get(c, UNK)
                        print(r)
                        pred_w = target_word_ids.get(r, UNK)
                        results_c.append(pred_w)
                    concat_pred_w[is_unk] = results_c
                result.append(concat_pred_w)
        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t[0].tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        reporter.report({self.key: bleu})


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        return {line.strip(): i for i, line in enumerate(f)}


def load_data(word_voc, char_voc, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = np.array([word_voc.get(w, UNK) for w in words], dtype=np.int32)
            unk_words = np.array(words)[array==UNK]
            unk_array = np.array([
                np.array([char_voc.get(c, UNK) for c in list(w)], dtype=np.int32)
                for w in unk_words])
            array = np.array([array, unk_array])
            if len(unk_array)!=0:
                print(array)
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s[0] == UNK).sum() for s in data)
    total = sum(s[0].size for s in data)
    return unknown / total


def main():
    global target_words, target_word_ids, target_chars, target_char_ids
    todaydetail = dt.today()
    todaydetailf = todaydetail.strftime("%Y%m%d-%H:%M:%S")
    print('start at ' + todaydetailf)
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_WORD_VOCAB', help='source word vocabulary file')
    parser.add_argument('TARGET_WORD_VOCAB', help='target word vocabulary file')
    parser.add_argument('SOURCE_CHAR_VOCAB', help='source char vocabulary file')
    parser.add_argument('TARGET_CHAR_VOCAB', help='target char vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--trigger', '-t', type=int, default=4000,
                        help='define trigger')
    args = parser.parse_args()

    source_word_ids = load_vocabulary(args.SOURCE_WORD_VOCAB)
    target_word_ids = load_vocabulary(args.TARGET_WORD_VOCAB)
    source_words = {i: w for w, i in source_word_ids.items()}
    target_words = {i: w for w, i in target_word_ids.items()}
    source_char_ids = load_vocabulary(args.SOURCE_CHAR_VOCAB)
    target_char_ids = load_vocabulary(args.TARGET_CHAR_VOCAB)
    #source_chars = {i: w for w, i in source_char_ids.items()}
    target_chars = {i: w for w, i in target_char_ids.items()}
    train_source = load_data(source_word_ids, source_char_ids, args.SOURCE)
    train_target = load_data(target_word_ids, target_char_ids, args.TARGET)
    assert len(train_source) == len(train_target)
    train_data = [(s, t)
                  for s, t in six.moves.zip(train_source, train_target)
                  if args.min_source_sentence <= len(s[0])
                  <= args.max_source_sentence and
                  args.min_source_sentence <= len(t[0])
                  <= args.max_source_sentence]
    train_source_unknown = calculate_unknown_ratio(
        [s for s, _ in train_data])
    train_target_unknown = calculate_unknown_ratio(
        [t for _, t in train_data])

    print('Source word vocabulary size: %d' % len(source_word_ids))
    print('Target word vocabulary size: %d' % len(target_word_ids))
    print('Source char vocabulary size: %d' % len(source_char_ids))
    print('Target char vocabulary size: %d' % len(target_char_ids))
    print('Train data size: %d' % len(train_data))
    print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))

    model = Seq2seq(args.layer, len(source_word_ids), len(target_word_ids), len(source_char_ids), len(target_char_ids), args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(args.trigger, 'iteration'), log_name='Log-'+todaydetailf+'.txt'),
                   trigger=(args.trigger, 'iteration'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'validation/main/bleu',
         'elapsed_time']),
        trigger=(200, 'iteration'))

    if args.validation_source and args.validation_target:
        test_source = load_data(source_word_ids, source_char_ids, args.validation_source)
        test_target = load_data(target_word_ids, target_char_ids, args.validation_target)
        assert len(test_source) == len(test_target)
        test_data = list(six.moves.zip(test_source, test_target))
        test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
        test_source_unknown = calculate_unknown_ratio(
            [s for s, _ in test_data])
        test_target_unknown = calculate_unknown_ratio(
            [t for _, t in test_data])

        print('Validation data: %d' % len(test_data))
        print('Validation source unknown ratio: %.2f%%' %
              (test_source_unknown * 100))
        print('Validation target unknown ratio: %.2f%%' %
              (test_target_unknown * 100))

        @chainer.training.make_extension(trigger=(200, 'iteration'))
        def translate(trainer):
            source, target = test_data[np.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            source_sentence = ' '.join([source_words[x] for x in source[0]])
            target_sentence = ' '.join([target_words[y] for y in target[0]])
            result_sentence = ' '.join([target_words[y] for y in result])
            print('# source : ' + source_sentence)
            print('#  result : ' + result_sentence)
            print('#  expect : ' + target_sentence)

        trainer.extend(translate, trigger=(args.trigger, 'iteration'))
        trainer.extend(
            CalculateBleu(
                model, test_data, 'validation/main/bleu', device=args.gpu),
            trigger=(args.trigger, 'iteration'))

    print('start training')
    trainer.run()
    print('=>finished!')
    model_name = todaydetailf+'-Hybrid-BiGRU.model'
    serializers.save_npz(cfg.PATH_TO_MODELS + model_name, model)
    print('=>save the model: '+model_name)
    enddetail = dt.today()
    enddetailf = enddetail.strftime("%Y%m%d-%H:%M:%S")
    print('end at ' + enddetailf)


if __name__ == '__main__':
    main()
