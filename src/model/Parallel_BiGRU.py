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
import my_chainer.links as My
from chainer import reporter
from chainer import training
from chainer.training import extensions
from chainer import serializers

UNK = 0
EOS = 1

target_words = {}
target_word_ids = {}
target_chars = {}
target_char_ids = {}
source_word_ids = {}
source_char_ids = {}

char_hidden = []

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs

class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_source_char, n_target_char, n_units):
        super(Seq2seq, self).__init__(
            embed_xw=L.EmbedID(n_source_vocab, n_units),
            embed_xc=L.EmbedID(n_source_char, n_units),
            embed_y=L.EmbedID(n_target_vocab, n_units * 2),
            encoder_fw=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            encoder_bw=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            encoder_fc=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            encoder_bc=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            decoder=My.NStepGRU(n_layers, n_units * 2, n_units * 2, 0.1),
            W=L.Linear(n_units * 2, n_target_vocab),
        )
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_params = 6
        
    def __call__(self, xs, ys):
        loss = self.CalcLoss(xs, ys)
        reporter.report({'loss': loss.data}, self)
        perp = self.xp.exp(loss.data)
        reporter.report({'perp': perp}, self)
        print("loss",loss)
        print()
        return loss
        
    def CalcLoss(self, xs, ys):
        wxs = [np.array([source_word_ids.get(w, UNK) for w in x], dtype=np.int32) for x in xs]
        cxs = [np.array([source_char_ids.get(c, UNK) for c in list("".join(x))], dtype=np.int32) for x in xs]
        concat_wxs = np.concatenate(wxs)
        concat_cxs = np.concatenate(cxs)
        
        # Target token can be either a word or a char
        wcys = [np.array([target_word_ids.get(w, UNK) for w in y], dtype=np.int32) for y in ys]
        
        eos = self.xp.array([EOS], 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in wcys]
        ys_out = [F.concat([y, eos], axis=0) for y in wcys]

        # Both xs and ys_in are lists of arrays.
        wexs = sequence_embed(self.embed_xw, wxs)
        cexs = sequence_embed(self.embed_xc, cxs)
        
        wexs_f = wexs
        wexs_b = [F.get_item(wex, range(len(wex))[::-1]) for wex in wexs]
        cexs_f = cexs
        cexs_b = [F.get_item(cex, range(len(cex))[::-1]) for cex in cexs]
        
        eys = sequence_embed(self.embed_y, ys_in)
        
        batch = len(xs)
        # None represents a zero vector in an encoder.
        _, hfw = self.encoder_fw(None, wexs_f)
        _, hbw = self.encoder_bw(None, wexs_b)
        _, hfc = self.encoder_fc(None, cexs_f)
        _, hbc = self.encoder_bc(None, cexs_b)
        
        # 隠れ状態ベクトルの集合
        hbw = [F.get_item(h, range(len(h))[::-1]) for h in hbw]
        hbc = [F.get_item(h, range(len(h))[::-1]) for h in hbc]
        htw = list(map(lambda x,y: F.concat([x, y], axis=1), hfw, hbw))
        htc = list(map(lambda x,y: F.concat([x, y], axis=1), hfc, hbc))
        ht = list(map(lambda x,y: F.concat([x, y], axis=0), htw, htc))
        h_list, h_bar_list, c_s_list, z_s_list = self.decoder(None, ht, eys)
        
        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        os = h_list
        os_len = [len(s) for s in os]
        os_section = np.cumsum(os_len[:-1])
        concat_os = F.concat(os, axis=0)
        concat_os_out = self.W(concat_os)
        concat_ys_out = F.concat(ys_out, axis=0)
        
        n_words = concat_ys_out.shape[0]
        loss = F.sum(F.softmax_cross_entropy(
            concat_os_out, concat_ys_out, reduce='no')) / n_words
        return loss

    def translate(self, xs, max_length=100):
        print("Now translating")
        batch = len(xs)
        print("batch",batch)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            wxs = [np.array([source_word_ids.get(w, UNK) for w in x], dtype=np.int32) for x in xs]
            wx_len = [len(wx) for wx in wxs]
            wx_section = np.cumsum(wx_len[:-1])
            valid_wx_section = np.insert(wx_section, 0, 0)
            cxs = [np.array([source_char_ids.get(c, UNK) for c in list("".join(x))], dtype=np.int32) for x in xs]
            
            wexs = sequence_embed(self.embed_xw, wxs)
            cexs = sequence_embed(self.embed_xc, cxs)
            
            wexs_f = wexs
            wexs_b = [wex[::-1] for wex in wexs]
            cexs_f = cexs
            cexs_b = [cex[::-1] for cex in cexs]
            
            _, hfw = self.encoder_fw(None, wexs_f)
            _, hbw = self.encoder_bw(None, wexs_b)
            _, hfc = self.encoder_fc(None, cexs_f)
            _, hbc = self.encoder_bc(None, cexs_b)
            
            hbw = [F.get_item(h, range(len(h))[::-1]) for h in hbw]
            hbc = [F.get_item(h, range(len(h))[::-1]) for h in hbc]
            htw = list(map(lambda x,y: F.concat([x, y], axis=1), hfw, hbw))
            htc = list(map(lambda x,y: F.concat([x, y], axis=1), hfc, hbc))
            ht = list(map(lambda x,y: F.concat([x, y], axis=0), htw, htc))
            
            ys = self.xp.full(batch, EOS, 'i')
            result = []
            h=None
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = chainer.functions.split_axis(eys, batch, 0)
                h_list, h_bar_list, c_s_list, z_s_list = self.decoder(h, ht, eys)
                cys = chainer.functions.concat(h_list, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)
                h = F.transpose_sequence(h_list)[-1]
                h = F.reshape(h, (self.n_layers, h.shape[0], h.shape[1]))
        
        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs
    
    def CalculateValLoss(self, xs, ys):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            loss = self.CalcLoss(xs, ys)
        return loss.data
    
    def get_n_params(self):
        return self.n_params

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
            for i in range(0, len(self.test_data[0:100]), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t[0].tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = self.model.translate(sources, self.max_length)
                ys = [y.tolist() for y in ys]
                hypotheses.extend(ys)
            
            source, target = zip(*self.test_data[0:100])
            loss = self.model.CalculateValLoss(source, target)
        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        reporter.report({self.key[0]: bleu})
        reporter.report({self.key[1]: loss})


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
            data.append(np.array(words))
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s[0] == UNK).sum() for s in data)
    total = sum(s[0].size for s in data)
    return unknown / total


def main():
    global target_words, target_word_ids, target_chars, target_char_ids,source_word_ids,source_char_ids
    todaydetail = dt.today()
    todaydetailf = todaydetail.strftime("%Y%m%d-%H%M%S")
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
                  if args.min_source_sentence <= len(s)
                  <= args.max_source_sentence and
                  args.min_source_sentence <= len(t)
                  <= args.max_source_sentence]
    #train_source_unknown = calculate_unknown_ratio(
    #    [s for s, _ in train_data])
    #train_target_unknown = calculate_unknown_ratio(
    #    [t for _, t in train_data])

    print('Source word vocabulary size: %d' % len(source_word_ids))
    print('Target word vocabulary size: %d' % len(target_word_ids))
    print('Source char vocabulary size: %d' % len(source_char_ids))
    print('Target char vocabulary size: %d' % len(target_char_ids))
    print('Train data size: %d' % len(train_data))
    #print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    #print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))

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
        ['epoch', 'iteration',
         'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'validation/main/bleu',
         'elapsed_time']),
        trigger=(args.trigger, 'iteration'))

    if args.validation_source and args.validation_target:
        test_source = load_data(source_word_ids, source_char_ids, args.validation_source)
        test_target = load_data(target_word_ids, target_char_ids, args.validation_target)
        assert len(test_source) == len(test_target)
        test_data = list(six.moves.zip(test_source, test_target))
        test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
        #test_source_unknown = calculate_unknown_ratio(
        #    [s for s, _ in test_data])
        #test_target_unknown = calculate_unknown_ratio(
        #    [t for _, t in test_data])

        print('Validation data: %d' % len(test_data))
        #print('Validation source unknown ratio: %.2f%%' %
        #      (test_source_unknown * 100))
        #print('Validation target unknown ratio: %.2f%%' %
        #      (test_target_unknown * 100))

        @chainer.training.make_extension(trigger=(args.trigger, 'iteration'))
        def translate(trainer):
            source, target = test_data[np.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            source_sentence = ' '.join([x for x in source])
            target_sentence = ' '.join([y for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            print('# source : ' + source_sentence)
            print('#  result : ' + result_sentence)
            print('#  expect : ' + target_sentence)

        #trainer.extend(translate, trigger=(args.trigger, 'iteration'))
        trainer.extend(
            CalculateBleu(
                model, test_data, 
                ['validation/main/bleu', 'validation/main/loss'],
                device=args.gpu),
            trigger=(args.trigger, 'iteration'))
    
    print('start training')
    trainer.run()
    print('=>finished!')
    
    model_name = todaydetailf+'-Hybrid-BiGRU.model'
    serializers.save_npz(cfg.PATH_TO_MODELS + model_name, model)
    print('=>save the model: '+model_name)
    
    config_name = todaydetailf+'-Hybrid-BiGRU-config.txt'
    f = open(cfg.PATH_TO_MODELS + config_name, 'w')
    model_params = [str(args.layer), str(len(source_word_ids)), str(len(target_word_ids)), str(len(source_char_ids)), str(len(target_char_ids)), str(args.unit)]
    assert len(model_params)==model.get_n_params()
    f.write("\n".join(model_params))
    f.close()
    print('=>save the config: '+config_name)
    
    enddetail = dt.today()
    enddetailf = enddetail.strftime("%Y%m%d-%H:%M:%S")
    print('end at ' + enddetailf)


if __name__ == '__main__':
    main()
