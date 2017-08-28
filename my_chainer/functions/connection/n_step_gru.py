import itertools

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.connection import n_step_rnn
from chainer.functions.connection.n_step_rnn import get_random_state
from chainer.functions.noise import dropout
from chainer.utils import argument


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


class NStepGRU(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, **kwargs):
        n_step_rnn.BaseNStepRNN.__init__(
            self, n_layers, states, rnn_dir='uni', rnn_mode='gru', **kwargs)


class NStepBiGRU(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, **kwargs):
        n_step_rnn.BaseNStepRNN.__init__(
            self, n_layers, states, rnn_dir='bi', rnn_mode='gru', **kwargs)


def n_step_gru(
        n_layers, dropout_ratio, hx, ws, bs, xs, W, B, **kwargs):
    """n_step_gru(n_layers, dropout_ratio, hx, ws, bs, xs)

    Stacked Uni-directional Gated Recurrent Unit function.

    This function calculates stacked Uni-directional GRU with sequences.
    This function gets an initial hidden state :math:`h_0`, an input
    sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` for each time :math:`t`
    from input :math:`x_t`.

    .. math::
       r_t &= \\sigma(W_0 x_t + W_3 h_{t-1} + b_0 + b_3) \\\\
       z_t &= \\sigma(W_1 x_t + W_4 h_{t-1} + b_1 + b_4) \\\\
       h'_t &= \\tanh(W_2 x_t + b_2 + r_t \\cdot (W_5 h_{t-1} + b_5)) \\\\
       h_t &= (1 - z_t) \\cdot h'_t + z_t \\cdot h_{t-1}

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Six weight matrices and six bias vectors are
    required for each layers. So, when :math:`S` layers exists, you need to
    prepare :math:`6S` weigth matrices and :math:`6S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    .. warning::

       ``train`` and ``use_cudnn`` arguments are not supported anymore since
       v2.
       Instead, use ``chainer.using_config('train', train)`` and
       ``chainer.using_config('use_cudnn', use_cudnn)`` respectively.
       See :func:`chainer.using_config`.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimention of
            hidden units.
        xs (list of chainer.Variable): A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this functions supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.

    Returns:
        tuple: This functions returns a tuple concaining three elements,
            ``hy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    """

    return n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, W, B,
                           use_bi_direction=False, **kwargs)


def n_step_bigru(
        n_layers, dropout_ratio, hx, ws, bs, xs, **kwargs):
    """n_step_bigru(n_layers, dropout_ratio, hx, ws, bs, xs)

    Stacked Bi-directional Gated Recurrent Unit function.

    This function calculates stacked Bi-directional GRU with sequences.
    This function gets an initial hidden state :math:`h_0`, an input
    sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` for each time :math:`t`
    from input :math:`x_t`.

    .. math::
       r^{f}_t &= \\sigma(W^{f}_0 x_t + W^{f}_3 h_{t-1} + b^{f}_0 + b^{f}_3)
       \\\\
       z^{f}_t &= \\sigma(W^{f}_1 x_t + W^{f}_4 h_{t-1} + b^{f}_1 + b^{f}_4)
       \\\\
       h^{f'}_t &= \\tanh(W^{f}_2 x_t + b^{f}_2 + r^{f}_t \\cdot (W^{f}_5
       h_{t-1} + b^{f}_5)) \\\\
       h^{f}_t &= (1 - z^{f}_t) \\cdot h^{f'}_t + z^{f}_t \\cdot h_{t-1}
       \\\\
       r^{b}_t &= \\sigma(W^{b}_0 x_t + W^{b}_3 h_{t-1} + b^{b}_0 + b^{b}_3)
       \\\\
       z^{b}_t &= \\sigma(W^{b}_1 x_t + W^{b}_4 h_{t-1} + b^{b}_1 + b^{b}_4)
       \\\\
       h^{b'}_t &= \\tanh(W^{b}_2 x_t + b^{b}_2 + r^{b}_t \\cdot (W^{b}_5
       h_{t-1} + b^{b}_5)) \\\\
       h^{b}_t &= (1 - z^{b}_t) \\cdot h^{b'}_t + z^{b}_t \\cdot h_{t-1}
       \\\\
       h_t  &= [h^{f}_t; h^{f}_t] \\\\

    where :math:`W^{f}` is weight matrices for forward-GRU, :math:`W^{b}` is
    weight matrices for backward-GRU.

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Six weight matrices and six bias vectors are
    required for each layers. So, when :math:`S` layers exists, you need to
    prepare :math:`6S` weigth matrices and :math:`6S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    .. warning::

       ``train`` and ``use_cudnn`` arguments are not supported anymore since
       v2.
       Instead, use ``chainer.using_config('train', train)`` and
       ``chainer.using_config('use_cudnn', use_cudnn)`` respectively.
       See :func:`chainer.using_config`.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimention of
            hidden units.
        xs (list of chainer.Variable): A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this functions supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        use_bi_direction (bool): If ``True``, this function uses
            Bi-direction GRU.

    Returns:
        tuple: This functions returns a tuple concaining three elements,
            ``hy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    """

    return n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs,
                           use_bi_direction=True, **kwargs)


def n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, W, B, 
                    use_bi_direction, **kwargs):
    """n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, use_bi_direction)

    Base function for Stack GRU/BiGRU functions.

    This function is used at  :func:`chainer.functions.n_step_bigru` and
    :func:`chainer.functions.n_step_gru`.
    This function's behavior depends on argument ``use_bi_direction``.

    .. warning::

       ``train`` and ``use_cudnn`` arguments are not supported anymore since
       v2.
       Instead, use ``chainer.using_config('train', train)`` and
       ``chainer.using_config('use_cudnn', use_cudnn)`` respectively.
       See :func:`chainer.using_config`.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimention of
            hidden units.
        xs (list of chainer.Variable): A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this functions supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.
        use_bi_direction (bool): If ``True``, this function uses
            Bi-direction GRU.

    .. seealso::
       :func:`chainer.functions.n_step_rnn`
       :func:`chainer.functions.n_step_birnn`

    """  # NOQA
    argument.check_unexpected_kwargs(
        kwargs, train='train argument is not supported anymore. '
        'Use chainer.using_config',
        use_cudnn='use_cudnn argument is not supported anymore. '
        'Use chainer.using_config')
    argument.assert_kwargs_empty(kwargs)

    #xp = cuda.get_array_module(hx, hx.data)

    if False:
    #if xp is not numpy and chainer.should_use_cudnn('>=auto', 5000):
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx, ),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            xs))
        if use_bi_direction:
            rnn = NStepBiGRU(n_layers, states)
        else:
            rnn = NStepGRU(n_layers, states)

        ret = rnn(*inputs)
        hy, = ret[:1]
        ys = ret[1:]
        return hy, ys

    else:
        direction = 2 if use_bi_direction else 1
        '''
        hx = split_axis.split_axis(hx, n_layers * direction, axis=0,
                                   force_tuple=True)
        hx = [reshape.reshape(h, h.shape[1:]) for h in hx]
        '''

        xws = [concat.concat([w[0], w[1], w[2]], axis=0) for w in ws]
        hws = [concat.concat([w[3], w[4], w[5]], axis=0) for w in ws]
        xbs = [concat.concat([b[0], b[1], b[2]], axis=0) for b in bs]
        hbs = [concat.concat([b[3], b[4], b[5]], axis=0) for b in bs]
        
        W1, W2, W3 = W
        B1, B2, B3 = B
        
        #hx:encoderの最上層の隠れ状態ベクトルの集合
        ht = hx
        '''
        print(type(ht), len(ht))
        print(type(ht[0]), len(ht[0]))
        print(type(ht[0][0]), len(ht[0][0]))
        
        print(type(W1), len(W1))
        print(type(W1[0]), len(W1[0]))
        '''
        phi_ht = list(map(lambda x: linear.linear(x, W1, B1), ht))

        xs_next = xs

        for layer in six.moves.range(n_layers):

            def _one_directional_loop(di):
                # di=0, forward GRU
                # di=1, backward GRU
                xs_list = xs_next if di == 0 else reversed(xs_next)
                layer_idx = direction * layer + di
                #h = hx[layer_idx]
                
                # h:d_bar_s_1
                # h_bar:d_s
                '''
                print(len(xs_list))
                print(len(xs_list[0]))
                print(len(xs_list[0][0]))
                '''
                h = np.zeros((xs_list[0].shape), dtype=np.float32)
                h_list = []
                h_bar_list = []
                c_s_list = []
                z_s_list = []
                for x in xs_list:
                    batch = x.shape[0]
                    
                    if h.shape[0] > batch:
                        h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                    else:
                        h_rest = None
                    
                    if layer > 0:
                        x = dropout.dropout(x, ratio=dropout_ratio)

                    gru_x = linear.linear(x, xws[layer_idx], xbs[layer_idx])
                    gru_h = linear.linear(h, hws[layer_idx], hbs[layer_idx])

                    W_r_x, W_z_x, W_x = split_axis.split_axis(gru_x, 3, axis=1)
                    U_r_h, U_z_h, U_x = split_axis.split_axis(gru_h, 3, axis=1)

                    r = sigmoid.sigmoid(W_r_x + U_r_h)
                    z = sigmoid.sigmoid(W_z_x + U_z_h)
                    h_bar = tanh.tanh(W_x + r * U_x)
                    h_bar = (1 - z) * h_bar + z * h
                    
                    phi_d = linear.linear(h_bar, W2, B2)
                    '''
                    print(type(phi_ht), len(phi_ht))
                    print(type(phi_ht[0]), len(phi_ht[0]))
                    print(type(phi_ht[0][0]), len(phi_ht[0][0]))
                    
                    print(type(phi_d), len(phi_d))
                    print(type(phi_d[0]), len(phi_d[0]), phi_d[0].shape)
                    '''
                    #phi_ht_len = [t.shape[1] for t in phi_ht]
                    #phi_ht_section = np.cumsum(phi_ht_len[:-1])
                    #concat_phi_ht  = F.concat(phi_ht, axis=1)
                    #concat_phi_d = [F.concat([phi_d[i]]*phi_ht_len[i], axis=0) for i in range(batch)]
                    #concat_phi_d = F.concat(concat_phi_d, axis=0)
                    #concat_phi_d = F.concat(F.transpose(phi_d), axis=0)
                    
                    u_st = list(map(lambda x,y: 
                                    (linear.linear(x, y.reshape(1,len(y)))).reshape(len(x),),
                                    phi_ht, phi_d))   #(4)
                    

                    sum_u = list(map(F.sum, u_st))
                    alpha_st = list(map(lambda x,y:x/F.broadcast_to(y, x.shape), u_st, sum_u))   #(3)
                    z_s = list(map(F.argmax, alpha_st))
                    z_s = list(map(lambda x:F.broadcast_to(x, (1,)), z_s))
                    z_s = F.concat(z_s, axis=0)
                    '''
                    print(type(alpha_st),len(alpha_st))
                    print(type(alpha_st[0]),len(alpha_st[0]))
                    
                    print(alpha_st[0].shape)
                    print(ht[0].shape)
                    '''
                    c_s = list(map(lambda x,y:
                                         F.sum(F.broadcast_to(x.reshape(x.shape[0],1),y.shape)*y,axis=0),
                                         alpha_st, ht))   #(2)
                    
                    c_s_2d = list(map(lambda x:x.reshape(1,len(x)), c_s))
                    concat_c_s = F.concat(c_s_2d, axis=0)
                    
                    c_s = list(map(lambda x:F.broadcast_to(x,(1,len(x))), c_s))
                    c_s = F.concat(c_s, axis=0)
                    '''
                    print(type(c_s), len(c_s))
                    print(type(c_s[0]), len(c_s[0]), c_s[0].shape)
                    '''
                    h = F.relu(linear.linear(F.concat([concat_c_s, h_bar], axis=1), W3, B3))
                    
                    h_list.append(h)
                    h_bar_list.append(h_bar)
                    c_s_list.append(c_s)
                    z_s_list.append(z_s)
                    
                    #単語数の違いを担保
                    if h_rest is not None:
                        h = concat.concat([h, h_rest], axis=0)
                        h_bar = concat.concat([h_bar, h_rest], axis=0)
                    
                return h_list, h_bar_list, c_s_list, z_s_list

            # Forward GRU
            hs, h_bars, c_ss, z_ss = _one_directional_loop(di=0)

            '''
            if use_bi_direction:
                # Backward GRU
                h, h_backward = _one_directional_loop(di=1)
                h_backward.reverse()
                # Concat
                xs_next = [concat.concat([hfi, hbi], axis=1) for (hfi, hbi) in
                           six.moves.zip(h_forward, h_backward)]
                hy.append(h)
            else:
                # Uni-directional GRU
                xs_next = h_forward
             '''

        #ys = xs_next
        #hy = stack.stack(hy)
        #return hy, tuple(ys)
        return hs, h_bars, c_ss, z_ss