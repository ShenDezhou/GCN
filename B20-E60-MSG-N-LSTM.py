import codecs
import os
import string

import numpy
from keras import regularizers
from keras.layers import Dense, Embedding, LSTM, CuDNNLSTM, SpatialDropout1D, Input, Bidirectional, Dropout, \
    BatchNormalization, Lambda, concatenate, Flatten
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from scipy import sparse

#               precision    recall  f1-score   support
#
#            B     0.9445    0.9460    0.9453     56882
#            M     0.6950    0.8280    0.7557     11479
#            E     0.9421    0.9359    0.9390     56882
#            S     0.9441    0.9061    0.9247     47490
#
#    micro avg     0.9239    0.9239    0.9239    172733
#    macro avg     0.8814    0.9040    0.8912    172733
# weighted avg     0.9270    0.9239    0.9249    172733
# {'mean_squared_error': 0.2586491465518497, 'mean_absolute_error': 0.27396197698378544, 'mean_absolute_percentage_error': 0.3323864857505891, 'mean_squared_logarithmic_error': 0.2666326968685906, 'squared_hinge': 0.2827528866772688, 'hinge': 0.27436352076398335, 'categorical_crossentropy': 0.3050300775957548, 'binary_crossentropy': 0.7499999871882543, 'kullback_leibler_divergence': 0.30747676168440974, 'poisson': 0.2897763648871911, 'cosine_proximity': 0.3213321868358391, 'sgd': 0.27380688950156684, 'rmsprop': 0.4363407859974404, 'adagrad': 0.5028908227192664, 'adadelta': 0.3134481079882679, 'adam': 0.342444794579377, 'adamax': 0.36860069757644914, 'nadam': 0.39635284171196516}



words = []
with codecs.open('plain/actor_dic.utf8', 'r', encoding='utf8') as fa:
    lines = fa.readlines()
    lines = [line.strip() for line in lines]
    words.extend(lines)

rxwdict = dict(zip(words,range(1, 1+len(words))))
rxwdict['\n'] =0


rydict = dict(zip(list("ABCDEFZ"), range(len("ABCDEFZ"))))
ytick = [0,18,32,263.5,1346,2321,244001]


def getYClass(y):
    r = 0
    for i in range(len(ytick)):
        if int(y) >= ytick[i]:
            return r
        else:
            r+=1
    assert r<len(ytick), (y,r)
    return r


batch_size = 20
nFeatures = 5
seqlen = 225#85
totallen = nFeatures+seqlen
word_size = 11
actors_size = 8380
Hidden = 150
Regularization = 1e-4
Dropoutrate = 0.2
learningrate = 0.2
Marginlossdiscount = 0.2
nState = 7
EPOCHS = 60
modelfile = os.path.basename(__file__).split(".")[0]

loss = "squared_hinge"
optimizer = "nadam"

sequence = Input(shape=(totallen,))
seqsa= Lambda(lambda x: x[:, 0:nFeatures])(sequence)
seqsb = Lambda(lambda x: x[:,  nFeatures:])(sequence)

network_emb  = sparse.load_npz("model/weibo_wembedding.npz").todense()
embedded = Embedding(len(words) + 1, word_size, embeddings_initializer=Constant(network_emb), input_length=seqlen, mask_zero=False, trainable=True)(seqsb)

networkcore_emb  = sparse.load_npz("model/weibo_coreembedding.npz").todense()
embeddedc = Embedding(len(words) + 1, actors_size, embeddings_initializer=Constant(networkcore_emb), input_length=seqlen, mask_zero=False, trainable=True)(seqsb)

concat = concatenate([embedded, embeddedc])

# dropout = Dropout(Dropoutrate)(embedded)
dropout = SpatialDropout1D(rate=Dropoutrate)(concat)
# blstm = Bidirectional(LSTM(Hidden, dropout=Dropoutrate, recurrent_dropout=Dropoutrate, return_sequences=False), merge_mode='sum')(dropout)
blstm = Bidirectional(CuDNNLSTM(Hidden, return_sequences=False), merge_mode='sum')(dropout)

concat = concatenate([seqsa, blstm])
# dropout = Dropout(Dropoutrate)(blstm)
batchNorm = BatchNormalization()(concat)
dense = Dense(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(batchNorm)
model = Model(input=sequence, output=dense)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

model.summary()
# model.save("keras/B20-E60-F1-PU-L-Bn-De.h5")

MODE = 1

if MODE == 1:
    with codecs.open('plain/movie_sentiments.utf8', 'r', encoding='utf8') as fs:
        with codecs.open('plain/movie_years.utf8', 'r', encoding='utf8') as fy:
            with codecs.open('weibo_dic/movie_ndic.utf8', 'r', encoding='utf8') as fd:
                with codecs.open('plain/movie_actornames.utf8', 'r', encoding='utf8') as fa:
                    with codecs.open('plain/movie_states.utf8', 'r', encoding='utf8') as ff:
                        ylines = fy.readlines()
                        slines = fs.readlines()
                        alines = fa.readlines()
                        dlines = fd.readlines()
                        flines = ff.readlines()
                        assert len(dlines) == len(alines) and len(alines) == len(slines) and len(slines) == len(ylines) and len(flines) == len(ylines)
                        X = []
                        print('process X list.')
                        counter = 0
                        for i in range(len(dlines)):
                            item = []
                            item.append(ylines[i].strip())
                            item.extend(slines[i].strip().split(','))
                            item = [int(i) for i in item]  # year, p, n
                            item.append(item[1] + item[2])  # total
                            item.append(item[1] / item[2] if item[2] != 0 else 0)  # PN-ratio
                            anames = alines[i].strip().split(',')
                            item.extend([rxwdict.get(name, 0) for name in anames])
                            # pad right '\n'
                            #print(len(item))
                            item.extend([0]*(totallen - len(item)))
                            assert len(item) == totallen,(len(item))
                            X.append(item)
                            if counter % 10000 == 0 and counter != 0:
                                print('.')
                        X = numpy.array(X)
                        print(X.shape)

                        y=[]
                        print('process y list.')
                        for line in flines:
                            line = line.strip()
                            yi = numpy.zeros((len("ABCDEFZ")), dtype=int)
                            yi[getYClass(line)] = 1
                            y.append(yi)
                        y = numpy.array(y)
                        print(y.shape)

                        history = model.fit(X, y, batch_size=batch_size, nb_epoch=EPOCHS, verbose=1)

                        model.save("keras/%s.h5"%modelfile)
                        print('FIN')

if MODE == 2:
    STATES = list("BMES")
    with codecs.open('plain/pku_test.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('baseline/pku_test_B20-E60-F1-PU-L-Bn-De_states.txt', 'w', encoding='utf8') as fl:
            model = load_model("keras/B20-E60-F1-PU-L-Bn-De.h5")
            model.summary()

            xlines = ft.readlines()
            X = []
            print('process X list.')
            counter = 0
            for line in xlines:
                line = line.replace(" ", "").strip()
                # X.append([getFeaturesDict(line, i) for i in range(len(line))])
                X.append([rxdict.get(e, 0) for e in list(line)])
                counter += 1
                if counter % 1000 == 0 and counter != 0:
                    print('.')
            print(len(X))
            X = pad_sequences(X, maxlen=maxlen, padding='pre', value=0)
            print(len(X), X.shape)
            yp = model.predict(X)
            print(yp.shape)
            for i in range(yp.shape[0]):
                sl = yp[i]
                lens = len(xlines[i].strip())
                for s in sl[-lens:]:
                    i = numpy.argmax(s)
                    fl.write(STATES[i])
                fl.write('\n')
            print('FIN')
            # for sl in yp:
            #     for s in sl:
            #         i = numpy.argmax(s)
            #         fl.write(STATES[i])
            #     fl.write('\n')
            # print('FIN')
