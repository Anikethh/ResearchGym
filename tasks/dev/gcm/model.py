import tensorflow as tf
import hashlib
import tensorflow_probability as tfp


def sg(x):
    return tf.stop_gradient(x)

def get_variable(shape, name):
    seed = tf.get_current_name_scope() + name
    seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) & (2**32-1)
    if len(shape) == 2:
        l = 12. / (shape[0] + shape[1])
        res = tf.Variable(tf.random.uniform(shape, -l, l, seed=seed), name=name)
        return res
    else:
        return tf.Variable(tf.zeros(shape), name=name)

def get_embedding(shape, name):
    seed = tf.get_current_name_scope() + name
    seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) & (2**32-1)
    res = tf.Variable(tf.random.normal(shape, stddev=0.001, seed=seed), name=name)
    return res

def pos_w(w, k=10.0):
    return tf.nn.softplus(w * k) / k

def ce_loss(y, y_pred):
    return tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1-y) * tf.math.log(1 - y_pred + 1e-5))

def mse_loss(y, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y))

class BaseModel(tf.Module):
    def __init__(self, dense_dim=33, r_dim=4, hidden_dim=16, is_binary=1, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        self.dense_dim = dense_dim
        self.r_dim = r_dim
        self.hidden_dim = hidden_dim
        self.is_binary = is_binary


class MlpModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            s = x @ self.wx + r @ self.wr + self.b
            s = tf.nn.tanh(s)
            s = s @ self.w1 + self.b1
            s = tf.nn.tanh(s)
            s = s @ self.w2 + self.b2
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class PosNNModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            s = x @ self.wx + r @ pos_w(self.wr) + self.b
            s = tf.nn.tanh(s)
            s = s @ pos_w(self.w1) + self.b1
            s = tf.nn.tanh(s)
            s = s @ pos_w(self.w2) + self.b2
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class MinmaxModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.bx = get_variable([hidden_dim], 'bx')
            self.wx1 = get_variable([hidden_dim, hidden_dim], 'wx1')
            self.bx1 = get_variable([hidden_dim], 'bx1')
            self.wx2 = get_variable([hidden_dim, 9], 'wx2')
            self.b2 = get_variable([9], 'b2')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 9], 'wr2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            x = x @ self.wx + self.bx
            xr = r @ pos_w(self.wr) + self.br + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            x = x @ self.wx1 + self.bx1
            xr = xr @ pos_w(self.wr1) + self.br1 + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            s = xr @ pos_w(self.wr2) + x @ self.wx2 + self.b2
            s = tf.reshape(s, [-1, 3, 3])
            s = tf.reduce_min(s, -1)
            s = tf.reduce_max(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class SmoothedMinmaxModel(BaseModel):
    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.beta = beta
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.bx = get_variable([hidden_dim], 'bx')
            self.wx1 = get_variable([hidden_dim, hidden_dim], 'wx1')
            self.bx1 = get_variable([hidden_dim], 'bx1')
            self.wx2 = get_variable([hidden_dim, 9], 'wx2')
            self.b2 = get_variable([9], 'b2')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 9], 'wr2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            x = x @ self.wx + self.bx
            xr = r @ pos_w(self.wr) + self.br + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            x = x @ self.wx1 + self.bx1
            xr = xr @ pos_w(self.wr1) + self.br1 + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            s = xr @ pos_w(self.wr2) + x @ self.wx2 + self.b2
            s = tf.reshape(s, [-1, 3, 3])
            s = - tf.reduce_logsumexp(-s * self.beta, -1) / self.beta
            s = tf.reduce_logsumexp(s * self.beta, -1) / self.beta
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class ConstrainedModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.bx = get_variable([hidden_dim], 'bx')
            self.wx1 = get_variable([hidden_dim, hidden_dim], 'wx1')
            self.bx1 = get_variable([hidden_dim], 'bx1')
            self.wx2 = get_variable([hidden_dim, 1], 'wx2')
            self.b2 = get_variable([1], 'b2')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 1], 'wr2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            f1 = lambda x: tf.nn.elu(x)
            f2 = lambda x: -tf.nn.elu(-x)
            f3 = lambda x: f2(tf.nn.relu(x)-1) + f1(-tf.nn.relu(-x)+1)
            t = int(self.hidden_dim/3)
            fr = lambda x: tf.concat([f1(x[:, :t]), f2(x[:, t:2*t]), f3(x[:, 2*t:])], -1)
            x = x @ self.wx + self.bx
            xr = r @ pos_w(self.wr) + self.br + x
            x = tf.nn.tanh(x)
            xr = fr(xr)
            x = x @ self.wx1 + self.bx1
            xr = xr @ pos_w(self.wr1) + self.br1 + x
            x = tf.nn.tanh(x)
            xr = fr(xr)
            s = x @ self.wx2 + xr @ pos_w(self.wr2) + self.b2
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class ScalableModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx10 = get_variable([dense_dim, hidden_dim], 'wx10')
            self.bx10 = get_variable([hidden_dim], 'bx10')
            self.wx11 = get_variable([hidden_dim, hidden_dim], 'wx11')
            self.bx11 = get_variable([hidden_dim], 'bx11')
            self.wx12 = get_variable([hidden_dim, 1], 'wx12')
            self.wx121 = get_variable([hidden_dim, hidden_dim], 'wx121')
            self.bx121 = get_variable([hidden_dim], 'bx121')
            self.wx20 = get_variable([dense_dim, hidden_dim], 'wx20')
            self.bx20 = get_variable([hidden_dim], 'bx20')
            self.wx21 = get_variable([hidden_dim, hidden_dim], 'wx21')
            self.wx22 = get_variable([hidden_dim, 1], 'wx22')
            self.b2 = get_variable([1], 'b2')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 1], 'wr2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            relun = lambda x: tf.nn.relu(x) - tf.nn.relu(x - 2)
            xr = r @ pos_w(self.wr) + self.br
            x1 = x @ self.wx10 + self.bx10
            x2 = x @ self.wx20 + self.bx20
            xr = relun(xr)
            x1 = tf.nn.relu(x1)
            x2 = relun(x2)
            xr = xr @ pos_w(self.wr1) + self.br1 + x2 @ self.wx21
            x2 = x1 @ self.wx121 + self.bx121
            x1 = x1 @ self.wx11 + self.bx11
            xr = relun(xr)
            x1 = tf.nn.relu(x1)
            x2 = relun(x2)
            s = x1 @ self.wx12 + x2 @ self.wx22 + xr @ pos_w(self.wr2) + self.b2
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss

class HintModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            s = x @ self.wx + r @ self.wr + self.b
            s = tf.tanh(s)
            s = s @ self.w1 + self.b1
            s = tf.tanh(s)
            s = s @ self.w2 + self.b2
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            t = tf.sign(tf.random.uniform([r.shape[0], 1], seed = self.seed)-0.5)
            self.seed += 1
            r_ = r + tf.random.uniform(r.shape, seed = self.seed) * 0.1 * t
            self.seed += 1
            s_ = x @ self.wx + r_ @ self.wr + self.b
            s_ = tf.tanh(s_)
            s_ = s_ @ self.w1 + self.b1
            s_ = tf.tanh(s_)
            s_ = s_ @ self.w2 + self.b2
            s_ = tf.squeeze(s_, -1)
            delta = tf.reduce_mean(tf.nn.relu((s-s_) * t) ** 2)
            delta -= sg(delta)
            loss += delta
            return y_pred, loss

class PwlModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    def __call__(self, inputs, y, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            h1 = x @ self.wx + r @ self.wr + self.b
            h1 = tf.tanh(h1)
            h2 = h1 @ self.w1 + self.b1
            h2 = tf.tanh(h2)
            s = h2 @ self.w2 + self.b2
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            h2_grads = (tf.transpose(self.w2) * (1-tf.square(sg(h2)))) @ tf.transpose(self.w1)
            r_grads = (h2_grads * (1-tf.square(sg(h1)))) @ tf.transpose(self.wr)
            reg = tf.reduce_sum(tf.nn.relu(-r_grads)) * 0.01
            reg -= sg(reg)
            loss += reg
        return y_pred, loss

afunc = lambda x: (1 + tf.nn.relu(x)) / (1 + tf.nn.relu(-x)) + 0.1

class GcmModel(BaseModel):
    def __init__(self, sample_num = 32, test_sample_num = 128, z_dim=4, loss_type=0, beta=1, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.loss_type = loss_type
            self.sample_num = sample_num
            self.test_sample_num = test_sample_num
            self.beta = beta
            self.test_z = tf.constant(tf.random.normal([test_sample_num, z_dim], seed=self.seed))
            self.seed += 1
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wmu = get_variable([hidden_dim, z_dim], 'wmu')
            self.bmu = get_variable([z_dim], 'bmu')
            self.wsig = get_variable([hidden_dim, z_dim], 'wsig')
            self.bsig = get_variable([z_dim], 'bsig')
            self.wh = get_variable([z_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wmc = get_variable([hidden_dim, r_dim], 'wmc')
            self.bmc = get_variable([r_dim], 'bmc')
            self.wsc = get_variable([hidden_dim, r_dim], 'wsc')
            self.bsc = get_variable([r_dim], 'bsc')
            self.wr = get_variable([r_dim, r_dim], 'wr')
            if loss_type == 2:
                self.wt1 = get_variable([z_dim, hidden_dim], 'wt1')
                self.bt1 = get_variable([hidden_dim], 'bt1')
                self.wt2 = get_variable([hidden_dim, 1], 'wt2')
                self.bt2 = get_variable([1], 'bt2')
                self.wsy1 = get_variable([z_dim, hidden_dim], 'wsy1')
                self.bsy1 = get_variable([hidden_dim], 'bsy1')
                self.wsy2 = get_variable([hidden_dim, 1], 'wsy2')
                self.bsy2 = get_variable([1], 'bsy2')
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        sample_num = self.sample_num if not is_test else self.test_sample_num
        with tf.name_scope(f'{self.name}_net'):
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            mu = h @ self.wmu + self.bmu  # [b, d]
            log_var = h @ self.wsig + self.bsig # [b, d]
            mu = tf.tile(tf.expand_dims(mu, 1), [1, sample_num, 1])
            sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * log_var), 1), [1, sample_num, 1])
            rnd = self.test_z if is_test else tf.random.normal(tf.shape(mu), seed=self.seed)
            self.seed += 1
            z = rnd * sig + mu  # [b, k, d]  z ~ p(z|x)
            log_p_z = tf.reduce_sum(tfp.distributions.Normal(loc=0, scale=1).log_prob(z), -1)  # [b, k], log p(z), z ~ q(z|x)
            log_q_z = tf.reduce_sum(tfp.distributions.Normal(loc=mu, scale=sig).log_prob(z), -1)  # [b, k], log q(z|x), z ~ q(z|x)
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)  # [b, k, d]
            mu_c = h @ self.wmc + self.bmc
            s_c = afunc(h @ self.wsc + self.bsc)
            s = (tf.expand_dims(r @ pos_w(self.wr), 1) - mu_c) / s_c  # [b, k, nr]
            prob = tf.nn.sigmoid(s)
            prob = tf.reduce_prod(prob, -1)  # [b, k]
            _y = tf.expand_dims(y, -1)
            if self.loss_type==0:
                y_pred = tf.reduce_mean(prob, 1)  # [b]
                log_p_y = _y * tf.math.log(prob + 1e-5) + (1 - _y) * tf.math.log(1 - prob + 1e-5) # [b, k]   # log p(y|x,r,z), z ~ q(z|x,r,y)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z  - log_q_z), -1) - tf.math.log(float(sample_num))  # [b]
                elb = tf.reduce_mean(elb)
                loss = -elb
            elif self.loss_type==1:
                tmp = tf.reduce_mean(prob, 1)  # [b]
                y_pred = (tmp - 0.1) / 0.8
                log_p_y = (_y * 0.8 + 0.1) * tf.math.log(prob + 1e-5) + (1 - (_y * 0.8 + 0.1)) * tf.math.log(1 - prob + 1e-5)  # [b, k]   # log p(y|x,r,z), z ~ q(z|x,r,y)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z  - log_q_z), -1) - tf.math.log(float(sample_num))  # [b]
                elb = tf.reduce_mean(elb)
                loss = -elb
            else:
                t = z @ self.wt1 + self.bt1
                t = tf.nn.tanh(t)
                t = t @ self.wt2 + self.bt2
                t = tf.squeeze(t, -1)  # [b, k]
                s_y = z @ self.wsy1 + self.bsy1
                s_y = tf.nn.tanh(s_y)
                s_y = s_y @ self.wsy2 + self.bsy2
                s_y = afunc(s_y)
                s_y = tf.squeeze(s_y, -1)  # [b, k]
                mu_y = t + tf.math.log((1e-5 + prob) / (1e-5 + 1 - prob)) * s_y # [b, k]
                y_pred = tf.reduce_mean(mu_y, -1)
                tmp = tf.nn.sigmoid((tf.expand_dims(y, -1) - mu_y) / s_y)
                log_p_y = tf.math.log((1e-5 + tmp) * (1e-5 + 1 - tmp) / s_y)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z  - log_q_z), -1) - tf.math.log(float(sample_num))
                elb = tf.reduce_mean(elb)
                loss = -elb
            return y_pred, loss

class IgcmModel(BaseModel):
    def __init__(self, sample_num = 32, test_sample_num = 128, z_dim=4, kern_dim=None, loss_type=0, beta=1, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        if kern_dim is None:
            kern_dim = r_dim
        with tf.name_scope(f'{self.name}_net'):
            self.loss_type = loss_type
            self.sample_num = sample_num
            self.test_sample_num = test_sample_num
            self.beta = beta
            self.test_z = tf.constant(tf.random.normal([test_sample_num, z_dim], seed=self.seed))
            self.seed += 1
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wmu = get_variable([hidden_dim, z_dim], 'wmu')
            self.bmu = get_variable([z_dim], 'bmu')
            self.wsig = get_variable([hidden_dim, z_dim], 'wsig')
            self.bsig = get_variable([z_dim], 'bsig')
            self.wh = get_variable([z_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wmc = get_variable([hidden_dim, kern_dim], 'wmc')
            self.bmc = get_variable([kern_dim], 'bmc')
            self.wsc = get_variable([hidden_dim, kern_dim], 'wsc')
            self.bsc = get_variable([kern_dim], 'bsc')
            self.wr2kern = get_variable([r_dim, hidden_dim], 'wr2kern')
            self.br2kern = get_variable([hidden_dim], 'br2kern')
            self.wr2kernmu = get_variable([hidden_dim, kern_dim], 'wr2kernmu')
            self.br2kernmu = get_variable([kern_dim], 'br2kernmu')
            self.wr2kernsig = get_variable([hidden_dim, kern_dim], 'wr2kernsig')
            self.br2kernsig = get_variable([kern_dim], 'br2kernsig')
            self.wkern2r = get_variable([kern_dim, r_dim], 'wkern2r')
            self.bkern2r = get_variable([r_dim], 'bkern2r')
            self.rsig = get_variable([r_dim], 'rsig')
            if loss_type == 2:
                self.wt1 = get_variable([z_dim, hidden_dim], 'wt1')
                self.bt1 = get_variable([hidden_dim], 'bt1')
                self.wt2 = get_variable([hidden_dim, 1], 'wt2')
                self.bt2 = get_variable([1], 'bt2')
                self.wsy1 = get_variable([z_dim, hidden_dim], 'wsy1')
                self.bsy1 = get_variable([hidden_dim], 'bsy1')
                self.wsy2 = get_variable([hidden_dim, 1], 'wsy2')
                self.bsy2 = get_variable([1], 'bsy2')
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        sample_num = self.sample_num if not is_test else self.test_sample_num
        with tf.name_scope(f'{self.name}_net'):
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            mu = h @ self.wmu + self.bmu  # [b, d]
            log_var = h @ self.wsig + self.bsig # [b, d]
            mu = tf.tile(tf.expand_dims(mu, 1), [1, sample_num, 1])
            sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * log_var), 1), [1, sample_num, 1])
            rnd = self.test_z if is_test else tf.random.normal(tf.shape(mu), seed=self.seed)
            self.seed += 1
            z = rnd * sig + mu  # [b, k, d]  z ~ p(z|x)
            log_p_z = tf.reduce_sum(tfp.distributions.Normal(loc=0, scale=1).log_prob(z), -1)  # [b, k], log p(z), z ~ q(z|x)
            log_q_z = tf.reduce_sum(tfp.distributions.Normal(loc=mu, scale=sig).log_prob(z), -1)  # [b, k], log q(z|x), z ~ q(z|x)
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)  # [b, k, d]
            mu_c = h @ self.wmc + self.bmc
            s_c = afunc(h @ self.wsc + self.bsc)
            # kern
            h_kern = r @ self.wr2kern + self.br2kern
            h_kern = tf.nn.tanh(h_kern)
            mu_kern = h_kern @ self.wr2kernmu + self.br2kernmu
            logvar_kern = h_kern @ self.wr2kernsig + self.br2kernsig
            sig_kern = tf.math.exp(0.5 * logvar_kern)
            kern = tf.random.normal(tf.shape(mu_kern), seed=self.seed)
            self.seed += 1
            kern = mu_kern + kern * sig_kern
            r_hat_mu = kern @ pos_w(self.wkern2r) + self.bkern2r
            r_hat_logsig = self.rsig
            r_hat_sig = tf.math.exp(r_hat_logsig)
            r_elb = tf.reduce_sum(- 0.5 * tf.square((r - r_hat_mu) / r_hat_sig) - r_hat_logsig, -1)  + 0.5 * tf.reduce_sum(1 + logvar_kern - mu_kern ** 2 - sig_kern ** 2, -1)
            r_elb = tf.reduce_mean(r_elb) * 0.01
            r_elb -= sg(r_elb)
            #
            s = (tf.expand_dims(mu_kern, 1) - mu_c) / s_c  # [b, k, nr]
            prob =tf.nn.sigmoid(s)
            prob = tf.reduce_prod(prob, -1)  # [b, k]
            _y = tf.expand_dims(y, -1)
            if self.loss_type==0:
                y_pred = tf.reduce_mean(prob, 1)  # [b]
                log_p_y = _y * tf.math.log(prob + 1e-5) + (1 - _y) * tf.math.log(1 - prob + 1e-5) # [b, k]   # log p(y|x,r,z), z ~ q(z|x,r,y)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z  - log_q_z), -1) - tf.math.log(float(sample_num))  # [b]
                elb = tf.reduce_mean(elb)
                loss = -elb - r_elb
            elif self.loss_type==1:
                tmp = tf.reduce_mean(prob, 1)  # [b]
                y_pred = (tmp - 0.1) / 0.8
                log_p_y = (_y * 0.8 + 0.1) * tf.math.log(prob + 1e-5) + (1 - (_y * 0.8 + 0.1)) * tf.math.log(1 - prob + 1e-5)  # [b, k]   # log p(y|x,r,z), z ~ q(z|x,r,y)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z  - log_q_z), -1) - tf.math.log(float(sample_num))  # [b]
                elb = tf.reduce_mean(elb)
                loss = -elb - r_elb
            else:
                t = z @ self.wt1 + self.bt1
                t = tf.nn.tanh(t)
                t = t @ self.wt2 + self.bt2
                t = tf.squeeze(t, -1)  # [b, k]
                s_y = z @ self.wsy1 + self.bsy1
                s_y = tf.nn.tanh(s_y)
                s_y = s_y @ self.wsy2 + self.bsy2
                s_y = afunc(s_y)
                s_y = tf.squeeze(s_y, -1)  # [b, k]
                mu_y = t + tf.math.log((1e-5 + prob) / (1e-5 + 1 - prob)) * s_y # [b, k]
                y_pred = tf.reduce_mean(mu_y, -1)
                tmp = tf.nn.sigmoid((tf.expand_dims(y, -1) - mu_y) / s_y)
                log_p_y = tf.math.log((1e-5 + tmp) * (1e-5 + 1 - tmp) / s_y)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z  - log_q_z), -1) - tf.math.log(float(sample_num))
                elb = tf.reduce_mean(elb)
                loss = -elb - r_elb
            return y_pred, loss
