import codecs
import numpy
from scipy.sparse import csr_matrix, save_npz

#6880 unknown

words = []
with codecs.open('../plain/actor_dic.utf8', 'r', encoding='utf8') as fa:
    lines = fa.readlines()
    lines = [line.strip() for line in lines]
    words.extend(lines)

rxwdict = dict(zip(words,range(1, 1+len(words))))
rxwdict['。'] = 0

oov = []

dims=11
embedding_matrix = numpy.zeros((len(words)+1, int(dims)))
with codecs.open('weibo_nactors.csv', 'r', encoding='utf8') as ff:
    lines = ff.readlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if 'verifyName' in line:
            continue
        word, coefs = line.split(',', maxsplit=1)
        if word in rxwdict.keys():
            embedding_matrix[rxwdict[word]] = numpy.fromstring(coefs, 'f', sep=',')
        else:
            npy = numpy.fromstring(coefs, 'f', sep=',')
            assert len(npy) == dims, (len(npy), word, npy)
            oov.append(npy)

# print('mean:', len(oov))
# mean = numpy.mean(oov)
print(embedding_matrix.shape)
zeroind = numpy.where(~embedding_matrix.any(axis=1))[0]
print(zeroind)

# embedding_matrix[zeroind] = mean

# normalize
embedding_matrix = embedding_matrix/embedding_matrix.max(axis=0)

sparsem = csr_matrix(embedding_matrix)
save_npz("weibo_wembedding.npz", matrix=sparsem)

zeroind = numpy.where(~embedding_matrix.any(axis=1))[0]
print(zeroind)