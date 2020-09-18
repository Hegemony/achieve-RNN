import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import zipfile

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_jay_lyrics():
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))   # 单词的列表
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])   # 单词和数字组合的字典
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]    # 将歌词的单词转换为数字（由上面的字典定义的）
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

'''
首先，我们读取周杰伦专辑歌词数据集：
'''
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
# print((corpus_indices, char_to_idx, idx_to_char, vocab_size))
'''
one-hot向量:
为了将词表示成向量输入到神经网络，一个简单的办法是使用one-hot向量。假设词典中不同字符的数量为N（即词典大小vocab_size），
每个字符已经同一个从0到N−1的连续整数值索引一一对应。如果一个字符的索引是整数i, 那么我们创建一个全0的长为N的向量，
并将其位置为ii的元素设成1。该向量就是对原字符的one-hot向量。下面分别展示了索引为0和2的one-hot向量，向量长度等于词典大小。
'''
def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    # print(x, x.shape, x.shape[0])
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

x = torch.tensor([0, 2])
# print(x.view(-1, 1))
print(one_hot(x, vocab_size))
print('-'*100)
'''
scatter(dim, index, src) 的参数有 3 个
dim：沿着哪个维度进行索引
index：用来scatter的元素索引
src：用来scatter的源元素，可以是一个标量或一个张量,将src的张量或者标量替换进去
'''

'''
我们每次采样的小批量的形状是(批量大小, 时间步数)。下面的函数将这样的小批量变换成数个可以输入进网络的形状为(批量大小, 词典大小)的矩阵，
矩阵个数等于时间步数。这样一个单词一个单词输入到网络中，对应一个batch一个batch输入，每个batch的每行对应一个单词的one-hot编码
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)
print(X)
inputs = to_onehot(X, vocab_size)   # 得到一个列表，列表元素为tensor
print(inputs)
print(len(inputs), inputs[0].shape)
print('-'*100)

'''
初始化模型参数
'''
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
        # torch.nn.Parameter将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面，
        # 成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

'''
定义模型
我们根据循环神经网络的计算表达式实现该模型。首先定义init_rnn_state函数来返回初始化的隐藏状态。
它返回由一个形状为(批量大小, 隐藏单元个数)的值为0的NDArray组成的元组。使用元组是为了更便于处理隐藏状态含有多个NDArray的情况。
'''
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

'''
下面的rnn函数定义了在一个时间步里如何计算隐藏状态和输出。这里的激活函数使用了tanh函数。3.8节（多层感知机）中介绍过，
当元素在实数域上均匀分布时，tanh函数值的均值为0。
'''
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
print(inputs)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)
print(outputs)
print('-'*100)

'''
定义预测函数：
以下函数基于前缀prefix（含有数个字符的字符串）来预测接下来的num_chars个字符。这个函数稍显复杂，
其中我们将循环神经单元rnn设置成了函数参数，这样在后面小节介绍其他循环神经网络时能重复使用这个函数。
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    # print(output)
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # print(X)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # print(Y)
        # print('!' * 100)
        # print(int(Y[0].argmax(dim=1).item()))
        # print('@' * 100)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))  # 找到列元素的最大值取出索引
            # print(output)
            # print('#' * 100)
    return ''.join([idx_to_char[i] for i in output])

'''
我们先测试一下predict_rnn函数。我们将根据前缀“分开”创作长度为10个字符（不考虑前缀长度）的一段歌词。
因为模型参数为随机值，所以预测结果也是随机的。
'''
# (outputs, state) = rnn(inputs, state, params)
# print(outputs)
# # 拼接之后形状为(num_steps * batch_size, vocab_size)
# outputs = torch.cat(outputs, dim=0)
# print(outputs, outputs.size())
print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            device, idx_to_char, char_to_idx))

'''
裁剪梯度:
循环神经网络中较容易出现梯度衰减或梯度爆炸。为了应对梯度爆炸，我们可以裁剪梯度（clip gradient）。假设我们把所有模型参数梯度的元素
拼接成一个向量 g，并设裁剪的阈值是θ。裁剪后的梯度min(θ/||g||, 1)g, 裁剪后的梯度的L2范数不超过θ。
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

'''
困惑度:
我们通常使用困惑度（perplexity）来评价语言模型的好坏。回忆一下3.4节（softmax回归）中交叉熵损失函数的定义。
困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，

最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；对应交叉熵为0
最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；对应交叉熵很大
基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。
显然，任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小vocab_size。
'''

'''
定义模型训练函数:
跟之前章节的模型训练函数相比，这里的模型训练函数有以下几点不同：

使用困惑度评价模型。
在迭代模型参数前裁剪梯度。
对时序数据采用不同采样方法将导致隐藏状态初始化的不同。相关讨论可参考6.3节（语言模型数据集（周杰伦专辑歌词））。
另外，考虑到后面将介绍的其他循环神经网络，为了更通用，这里的函数实现更长一些。
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:    # 随机采样
        data_iter_fn = d2l.data_iter_random
    else:                 # 相邻采样
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)  # 加载数据，构造迭代器
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # print(outputs.size())
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)  # 把tensor变成在内存中连续分布的形式
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

'''
训练模型并创作歌词:
现在我们可以训练模型了。首先，设置模型超参数。我们将根据前缀“分开”和“不分开”分别创作长度为50个字符（不考虑前缀长度）的一段歌词。
我们每过50个迭代周期便根据当前训练的模型创作一段歌词。
'''
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['千里之外', '时代']
'''
下面采用随机采样训练模型并创作歌词。
'''
# train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
#                       vocab_size, device, corpus_indices, idx_to_char,
#                       char_to_idx, True, num_epochs, num_steps, lr,
#                       clipping_theta, batch_size, pred_period, pred_len,
#                       prefixes)

'''
接下来采用相邻采样训练模型并创作歌词。
'''
# train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
#                       vocab_size, device, corpus_indices, idx_to_char,
#                       char_to_idx, False, num_epochs, num_steps, lr,
#                       clipping_theta, batch_size, pred_period, pred_len,
#                       prefixes)

'''
对上面部分代码的解读
'''
print('*'*100)
is_random_iter = True
data_iter_fn = d2l.data_iter_consecutive
data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)  # 加载数据，构造迭代器
for X, Y in data_iter:
    if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
        state = init_rnn_state(batch_size, num_hiddens, device)
    else:
        for s in state:
            s.detach_()
    print(X, X.size())   # 32*35
    inputs = to_onehot(X, vocab_size)
    print(inputs, len(inputs), inputs[0].size())  # 35,  32*1027, 时间步为35，batch为32，词汇量为1027，35个32*1027的列表
    # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
    (outputs, state) = rnn(inputs, state, params)
    # 拼接之后形状为(num_steps * batch_size, vocab_size)
    outputs = torch.cat(outputs, dim=0)
    # Y的形状是(batch_size, num_steps)，转置后再变成长度为
    # batch * num_steps 的向量，这样跟输出的行一一对应
    print(Y, Y.size())          # 32*35
    print('-' * 100)
    print(torch.transpose(Y, 0, 1), torch.transpose(Y, 0, 1).size())  # 35*32
    print('-' * 100)
    print(torch.transpose(Y, 0, 1).contiguous(), torch.transpose(Y, 0, 1).contiguous().size())   # 35*32
    y = torch.transpose(Y, 0, 1).contiguous().view(-1)
    print('-' * 100)
    print(y, y.size())     # 1120
    print(y.long())
    print(outputs, outputs.size())   # 1120*1027
    print('='*100)

# import math
# 验证交叉熵函数
# entroy=nn.CrossEntropyLoss()
# input=torch.Tensor([[-0.7715, -0.6205, -0.2562],
#                     [-0.7715, -0.6205, -0.2562]])
# target = torch.tensor([0, 1])
# output = entroy(input, target)
# print(output)
# #根据公式计算