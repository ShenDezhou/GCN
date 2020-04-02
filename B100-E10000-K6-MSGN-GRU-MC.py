import codecs
import os
import string

import numpy
from keras import regularizers
from keras.layers import Dense, Embedding, LSTM, CuDNNLSTM, SpatialDropout1D, Input, Bidirectional, Dropout, \
    BatchNormalization, Lambda, concatenate, Flatten, Conv1D, CuDNNGRU, MaxPooling1D
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from scipy import sparse

#               precision    recall  f1-score   support
#
#            A     0.6667    0.1538    0.2500       130
#            B     0.5217    0.9231    0.6667       130
#
#    micro avg     0.5385    0.5385    0.5385       260
#    macro avg     0.5942    0.5385    0.4583       260
# weighted avg     0.5942    0.5385    0.4583       260
# {'mean_squared_error': 0.2586491465518497, 'mean_absolute_error': 0.27396197698378544, 'mean_absolute_percentage_error': 0.3323864857505891, 'mean_squared_logarithmic_error': 0.2666326968685906, 'squared_hinge': 0.2827528866772688, 'hinge': 0.27436352076398335, 'categorical_crossentropy': 0.3050300775957548, 'binary_crossentropy': 0.7499999871882543, 'kullback_leibler_divergence': 0.30747676168440974, 'poisson': 0.2897763648871911, 'cosine_proximity': 0.3213321868358391, 'sgd': 0.27380688950156684, 'rmsprop': 0.4363407859974404, 'adagrad': 0.5028908227192664, 'adadelta': 0.3134481079882679, 'adam': 0.342444794579377, 'adamax': 0.36860069757644914, 'nadam': 0.39635284171196516}



words = []
with codecs.open('plain/actor_dic.utf8', 'r', encoding='utf8') as fa:
    lines = fa.readlines()
    lines = [line.strip() for line in lines]
    words.extend(lines)

rxwdict = dict(zip(words,range(1, 1+len(words))))
rxwdict['\n'] =0



rydict = dict(zip(list("ABCDEF"), range(len("ABCDEF"))))
# ytick = [0,18,32,263.5,1346,2321,244001]
#ytick = [0, 2**0,2**4,2**5,2**8,2**10,2**11,2**18]
ytick = [0]
ytick.extend(numpy.logspace(0,5,num=5, base=10))
assert len(ytick) == 6, len(ytick)
STATES = list("ABCDEF")
# ytick = [0, 263.5, 244001]

def getYClass(y):
    r = 0
    for i in range(len(ytick)-1):
        if int(y) >= ytick[i] and int(y)<=ytick[i+1]:
            return r
        r+=1
    assert r<len(ytick), (y,r)
    return r



batch_size = 100
nFeatures = 5
seqlen = 225#85
totallen = nFeatures+seqlen
word_size = 11
actors_size=8380
Hidden = 150
nfilters = 150
kernelSize = 3
Regularization = 1e-4
Dropoutrate = 0.2
learningrate = 0.2
Marginlossdiscount = 0.2

nState = 6
EPOCHS = 100

modelfile = os.path.basename(__file__).split(".")[0]

loss = "squared_hinge"
optimizer = "nadam"

sequence = Input(shape=(totallen,))
seqsa= Lambda(lambda x: x[:, 0:5])(sequence)
seqsb = Lambda(lambda x: x[:,  5:])(sequence)
seqsc = Lambda(lambda x: x[:,  5:])(sequence)

network_emb  = sparse.load_npz("model/weibo_wembedding.npz").todense()
embedded = Embedding(len(words) + 1, word_size, embeddings_initializer=Constant(network_emb), input_length=seqlen, mask_zero=False, trainable=True)(seqsb)

networkcore_emb  = sparse.load_npz("model/weibo_coreembedding.npz").todense()
embeddedc = Embedding(len(words) + 1, actors_size, embeddings_initializer=Constant(networkcore_emb), input_length=seqlen, mask_zero=False, trainable=True)(seqsc)


dropout = Dropout(rate=Dropoutrate)(seqsa)
middle = Dense(Hidden, activation='relu', kernel_regularizer=regularizers.l2(Regularization))(dropout)
batchNorm = BatchNormalization()(middle)

dropoutb = SpatialDropout1D(rate=Dropoutrate)(embedded)
blstm = Bidirectional(CuDNNGRU(Hidden, return_sequences=False), merge_mode='sum')(dropoutb)
batchNormb = BatchNormalization()(blstm)

dropoutc = SpatialDropout1D(rate=Dropoutrate)(embeddedc)
conv = Conv1D(filters=nfilters, kernel_size=kernelSize)(dropoutc)
mpool = MaxPooling1D()(conv)
conv = Conv1D(filters=nfilters, kernel_size=kernelSize)(mpool)
mpool = MaxPooling1D()(conv)
conv = Conv1D(filters=nfilters, kernel_size=kernelSize)(mpool)
mpool = MaxPooling1D()(conv)
conv = Conv1D(filters=nfilters, kernel_size=kernelSize)(mpool)
mpool = MaxPooling1D()(conv)
conv = Conv1D(filters=nfilters, kernel_size=kernelSize)(mpool)
mpool = MaxPooling1D()(conv)
# blstmc = Bidirectional(CuDNNLSTM(Hidden, return_sequences=False), merge_mode='sum')(mpool)
batchNormc = BatchNormalization()(mpool)
flatten = Flatten()(batchNormc)
concat = concatenate([batchNorm, batchNormb, flatten])

dense = Dense(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(concat)
model = Model(input=sequence, output=dense)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

model.summary()
# model.save("keras/%s.h5"%modelfile)

MODE = 1

if MODE == 1:
    with codecs.open('plain/msg_training.utf8', 'r', encoding='utf8') as fx:
        with codecs.open('plain/msg_training_states.utf8', 'r', encoding='utf8') as fy:
            xlines = fx.readlines()
            ylines = fy.readlines()
            assert len(xlines) == len(ylines)
            X = []
            print('process X list.')
            counter = 0
            for i in range(len(xlines)):
                line = xlines[i].strip()
                segs = line.split(",")
                item = []
                sents = [float(s) for s in segs[0:5]]
                item.extend(sents)
                anames = segs[5:]
                item.extend([0] * (totallen - len(item) - len(anames)))
                item.extend([rxwdict.get(name, 0) for name in anames])
                # pad right '\n'
                # print(len(item))
                assert len(item) == totallen,(len(item))
                X.append(item)
                if counter % 1000 == 0 and counter != 0:
                    print('.')
            X = numpy.array(X)
            print(X.shape)

            y=[]
            print('process y list.')
            for line in ylines:
                line = line.strip()
                yi = numpy.zeros((len(STATES)), dtype=int)
                yi[getYClass(line)] = 1
                y.append(yi)
            y = numpy.array(y)
            print(y.shape)

            history = model.fit(X, y, batch_size=batch_size, nb_epoch=EPOCHS, verbose=1)
            model.save("keras/%s.h5"%modelfile)
            print('FIN')

# if MODE == 2:
    with codecs.open('plain/msg_test.utf8', 'r', encoding='utf8') as fx:
        with codecs.open('plain/msg_test_states.utf8', 'r', encoding='utf8') as fy:
            with codecs.open('baseline/msg_test_%s_states.utf8'%modelfile, 'w', encoding='utf8') as fp:
                model = load_model("keras/%s.h5"%modelfile)
                model.summary()

                xlines = fx.readlines()
                X = []
                print('process X list.')
                counter = 0
                for i in range(len(xlines)):
                    line = xlines[i].strip()
                    segs = line.split(",")
                    item = []
                    sents = [float(s) for s in segs[0:5]]
                    item.extend(sents)
                    anames = segs[5:]
                    item.extend([0] * (totallen - len(item) - len(anames)))
                    item.extend([rxwdict.get(name, 0) for name in anames])
                    assert len(item) == totallen, (len(item))
                    X.append(item)
                    if counter % 1000 == 0 and counter != 0:
                        print('.')
                    counter+=1
                X = numpy.array(X)
                print(X.shape)

                yp = model.predict(X)
                print(yp.shape)
                for i in range(yp.shape[0]):
                    i = numpy.argmax(yp[i])
                    fp.write(STATES[i])
                    fp.write('\n')
                print('FIN')
