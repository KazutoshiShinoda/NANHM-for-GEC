#!/usr/bin/env python

import argparse

from nltk.translate import bleu_score
import numpy as np
import progressbar
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from my_chainer.links.connection import n_step_gru as My
#from my_chainer.links.connection import att_gru_decoder as D
from chainer import reporter
from chainer import training
from chainer.training import extensions

#word-level
UNK = 0
EOS = 1

#char-level
BOW = 0   # Boundary 0f Words


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_source_char, n_target_char, n_units):
        super(Seq2seq, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units),
            embed_y=L.EmbedID(n_target_vocab, n_units * 2),
            embed_xc=L.EmbedID(n_source_char, n_units),
            embed_yc=L.EmbedID(n_target_char, n_units * 2),
            encoder_f=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            encoder_b=L.NStepGRU(n_layers, n_units, n_units, 0.1),
            decoder=My.NStepGRU(n_layers, n_units * 2, n_units * 2, 0.1),
            #decoder_att=D.AttGRUdec(n_layers, n_units * 2, n_units * 2, n_target_vocab),
            W=L.Linear(n_units * 2, n_target_vocab),
        )
        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        xs_f = xs
        xs_b = [x[::-1] for x in xs]

        eos = self.xp.array([EOS], 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs_f = sequence_embed(self.embed_x, xs_f)
        exs_b = sequence_embed(self.embed_x, xs_b)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        _, hf = self.encoder_f(None, exs_f)
        _, hb = self.encoder_b(None, exs_b)
        #hb = hb[::-1] <- するとエラー。しないのが正解？
        # 隠れ状態ベクトルの集合
        ht = list(map(lambda x,y: F.concat([x, y], axis=1), hf, hb))
        '''
        print(len(ht))
        print(len(ht[0]))
        print(len(ht[0][0]))
        '''
        h_list, h_bar_list, c_s_list, z_s_list = self.decoder(ht, eys)
        
        '''
        print("os")
        print(len(os))
        print(len(os[0]), len(os[1]))
        print(len(os[0][0]))
        '''
        
        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        os = h_list
        '''
        print(len(os), len(os[0]), len(os[0][0]))
        print(len(ys_out),len(ys_out[0]))
        '''
        concat_os = F.concat(os, axis=0)
        concat_os_out = self.W(concat_os)
        concat_ys_out = F.concat(ys_out, axis=0)
        '''
        print(concat_os_out.shape)
        print(concat_ys_out.shape)
        '''
        # If predicted word is UNK
        pred_w = F.argmax(concat_os_out, axis=1)
        UNKList = map(lambda x: x, pred_w)
        
        loss = F.sum(F.softmax_cross_entropy(
            concat_os_out, concat_ys_out, reduce='no')) / batch

        reporter.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        reporter.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs_f = xs
            xs_b = [x[::-1] for x in xs]
            exs_f = sequence_embed(self.embed_x, xs_f)
            exs_b = sequence_embed(self.embed_x, xs_b)
            _, hf = self.encoder_f(None, exs_f)
            _, hb = self.encoder_b(None, exs_b)
            ht = list(map(lambda x,y: F.concat([x, y], axis=1), hf, hb))
            ys = self.xp.full(batch, EOS, 'i')
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = chainer.functions.split_axis(eys, batch, 0)
                h_list, h_bar_list, c_s_list, z_s_list = self.decoder(ht, eys)
                cys = chainer.functions.concat(h_list, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)

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
                references.extend([[t.tolist()] for t in targets])

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


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = np.array([vocabulary.get(w, UNK) for w in words], 'i')
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def main():
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
    source_char_ids = load_vocabulary(args.SOURCE_CHAR_VOCAB)
    target_char_ids = load_vocabulary(args.TARGET_CHAR_VOCAB)
    train_source = load_data(source_word_ids, args.SOURCE)
    train_target = load_data(target_word_ids, args.TARGET)
    assert len(train_source) == len(train_target)
    train_data = [(s, t)
                  for s, t in six.moves.zip(train_source, train_target)
                  if args.min_source_sentence <= len(s)
                  <= args.max_source_sentence and
                  args.min_source_sentence <= len(t)
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

    target_words = {i: w for w, i in target_word_ids.items()}
    source_words = {i: w for w, i in source_word_ids.items()}

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
    trainer.extend(extensions.LogReport(trigger=(200, 'iteration')),
                   trigger=(200, 'iteration'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'validation/main/bleu',
         'elapsed_time']),
        trigger=(200, 'iteration'))

    if args.validation_source and args.validation_target:
        test_source = load_data(source_word_ids, args.validation_source)
        test_target = load_data(target_word_ids, args.validation_target)
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

            source_sentence = ' '.join([source_words[x] for x in source])
            target_sentence = ' '.join([target_words[y] for y in target])
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


if __name__ == '__main__':
    main()
