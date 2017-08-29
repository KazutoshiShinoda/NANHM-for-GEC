import numpy
import six

from chainer import cuda
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from my_chainer.functions.connection import n_step_gru as rnn
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.n_step_rnn import argsort_list_descent
from chainer.links.connection.n_step_rnn import permutate_list
from chainer.utils import argument
from chainer import variable


class NStepGRUBase(link.ChainList):

    """__init__(self, n_layers, in_size, out_size, dropout, use_bi_direction)

    Base link class for Stacked GRU/BiGRU links.

    This link is base link class for :func:`chainer.links.NStepRNN` and
    :func:`chainer.links.NStepBiRNN`.
    This link's behavior depends on argument, ``use_bi_direction``.

    .. warning::

       ``use_cudnn`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('use_cudnn', use_cudnn)``.
       See :func:`chainer.using_config`.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        use_bi_direction (bool): if ``True``, use Bi-directional GRU.
            if ``False``, use Uni-directional GRU.
    .. seealso::
        :func:`chainer.links.NStepGRU`
        :func:`chainer.links.NStepBiGRU`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, use_bi_direction,
                 **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, use_cudnn='use_cudnn argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        weights = []
        direction = 2 if use_bi_direction else 1
        for i in six.moves.range(n_layers):
            for di in six.moves.range(direction):
                weight = link.Link()
                with weight.init_scope():
                    for j in six.moves.range(6):
                        if i == 0 and j < 3:
                            w_in = in_size
                        elif i > 0 and j < 3:
                            w_in = out_size * direction
                        else:
                            w_in = out_size
                        w = variable.Parameter(
                            normal.Normal(numpy.sqrt(1. / w_in)),
                            (out_size, w_in))
                        b = variable.Parameter(0, (out_size,))
                        setattr(weight, 'w%d' % j, w)
                        setattr(weight, 'b%d' % j, b)
                    
                    w_in = out_size
                    w = variable.Parameter(
                        normal.Normal(numpy.sqrt(1. / w_in)),
                        (out_size, w_in))
                    b = variable.Parameter(0, (out_size,))
                    setattr(weight, 'w%d' % 6, w)
                    setattr(weight, 'b%d' % 6, b)
                    
                    w_in = out_size
                    w = variable.Parameter(
                        normal.Normal(numpy.sqrt(1. / w_in)),
                        (out_size, w_in))
                    b = variable.Parameter(0, (out_size,))
                    setattr(weight, 'w%d' % 7, w)
                    setattr(weight, 'b%d' % 7, b)
                    
                    w_in = out_size * 2
                    w = variable.Parameter(
                        normal.Normal(numpy.sqrt(1. / w_in)),
                        (out_size, w_in))
                    b = variable.Parameter(0, (out_size,))
                    setattr(weight, 'w%d' % 8, w)
                    setattr(weight, 'b%d' % 8, b)
                    
                weights.append(weight)

        super(NStepGRUBase, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout = dropout
        self.out_size = out_size
        self.direction = direction
        self.rnn = rnn.n_step_bigru if use_bi_direction else rnn.n_step_gru

    def init_hx(self, xs):
        shape = (self.n_layers * self.direction, len(xs), self.out_size)
        with cuda.get_device_from_id(self._device_id):
            hx = variable.Variable(self.xp.zeros(shape, dtype=xs[0].dtype))
        return hx

    def __call__(self, h0, hx, xs, **kwargs):
        """__call__(self, hx, xs)

        Calculate all hidden states and cell states.

        .. warning::

           ``train`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.

        Args:
            hx (~chainer.Variable or None): Initial hidden states. If ``None``
                is specified zero-vector is used.
            xs (list of ~chianer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.

        """
        argument.check_unexpected_kwargs(
            kwargs, train='train argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        assert isinstance(xs, (list, tuple)), "xs in not a list or tupple: %r" %type(xs)
        indices = argsort_list_descent(xs)

        xs = permutate_list(xs, indices, inv=False)
        
        if h0 is None:
            h0 = self.init_hx(xs)
        else:
            h0 = permutate_list(h0, indices, axis=1, inv=False)
        
        hx = permutate_list(hx, indices, inv=False)
        
        trans_x = transpose_sequence.transpose_sequence(xs)
        
        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5] for w in self]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5] for w in self]
        for w in self:
            w1 = w.w6
            w2 = w.w7
            w3 = w.w8
            b1 = w.b6
            b2 = w.b7
            b3 = w.b8
        W = [w1, w2, w3]
        B = [b1, b2, b3]

        h_list, h_bar_list, c_s_list, z_s_list = self.rnn(
            self.n_layers, self.dropout, h0, hx, ws, bs, trans_x, W, B)
        '''
        print(type(h_list),len(h_list))
        print(type(h_list[0]),len(h_list[0]))
        print(type(h_list[0][0]),len(h_list[0][0]))
        '''
        
        #batch内で入れ替え
        h_list = transpose_sequence.transpose_sequence(h_list)
        h_list = permutate_list(h_list, indices, inv=True)
        '''
        print(type(h_list),len(h_list))
        print(type(h_list[0]),len(h_list[0]))
        print(type(h_list[0][0]),len(h_list[0][0]))
        '''
        h_bar_list = transpose_sequence.transpose_sequence(h_bar_list)
        h_bar_list = permutate_list(h_bar_list, indices, inv=True)
        '''
        print(type(c_s_list),len(c_s_list))
        print(type(c_s_list[0]),len(c_s_list[0]))
        print(type(c_s_list[0][0]),len(c_s_list[0][0]), c_s_list[0][0])
        
        print(type(z_s_list), len(z_s_list))
        print(type(z_s_list[0]), len(z_s_list[0]), z_s_list[0])
        '''
        c_s_list = transpose_sequence.transpose_sequence(c_s_list)
        c_s_list = permutate_list(c_s_list, indices, inv=True)
        z_s_list = transpose_sequence.transpose_sequence(z_s_list)
        z_s_list = permutate_list(z_s_list, indices, inv=True)
        
        return h_list, h_bar_list, c_s_list, z_s_list


class NStepGRU(NStepGRUBase):

    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Uni-directional GRU for sequnces.

    This link is stacked version of Uni-directional GRU for sequences.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_gru`, this function automatically
    sort inputs in descending order by length, and transpose the sequence.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    .. warning::

       ``use_cudnn`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('use_cudnn', use_cudnn)``.
       See :func:`chainer.using_config`.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.

    .. seealso::
        :func:`chainer.functions.n_step_gru`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepGRUBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=False, **kwargs)


class NStepBiGRU(NStepGRUBase):

    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Bi-directional GRU for sequnces.

    This link is stacked version of Bi-directional GRU for sequences.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_bigru`, this function automatically
    sort inputs in descending order by length, and transpose the sequence.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    .. warning::

       ``use_cudnn`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('use_cudnn', use_cudnn)``.
       See :func:`chainer.using_config`.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.

    .. seealso::
        :func:`chainer.functions.n_step_bigru`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepGRUBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=True, **kwargs)