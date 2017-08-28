import chainer
import chainer.functions as F
import chainer.links as L
from chainer import link

att_size = 128

class AttGRUdec(link.ChainList):
    def __init__(self, n_layers, in_size, out_size, n_target_vocab):
        super(AttGRUdec, self).__init__(
            W1 = L.Linear(in_size, att_size),
            W2 = L.Linear(in_size, att_size),
            W3 = L.Linear(out_size * 2, out_size),
            rnn = L.StatelessGRU(in_size, out_size),
        )
        self.in_size = in_size
    def __call__(self, ht, xs, d_bar_s_1):
        #ht:encoderの最上層での隠れ状態ベクトルの集合
        #batch_size * n_words * in_size
        #xs:入力？答え？
        if d_bar_s_1 == None:
            d_bar_s_1 = np.zeros(self.in_size)
        
        ht_T = list(map(F.transpose, ht))
        phi_ht = list(map(W1, ht_T))
        
        d_s = rnn(d_bar_s_1, y_s_1)
        
        phi_d = F.transpose_sequence(W2(F.transpose_sequence(d_s)))
        u_st = list(map(lambda x: phi_d*x, phi_ht))   #(4)
        
        sum_u = F.sum(u_st)
        alpha_st = list(map(lambda x:x/sum_u, u_st))   #(3)
        z_s = F.argmax(alpha_st, axis=0)
        
        c_s = F.sum(list(map(lambda x,y:x*y , alpha_st, ht)))   #(2)
        
        d_bar_s = F.relu(W3(F.concat([c_s, d_s])))
        
        return d_bar_s, d_s, c_s, z_s