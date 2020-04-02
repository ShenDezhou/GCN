import codecs
import numpy

ytick = [0]
ytick.extend(numpy.logspace(0,5,num=5, base=10))
assert len(ytick) == 6, len(ytick)
STATES = list("ABCDEF")

def getYClass(y):
    r = 0
    for i in range(len(ytick)-1):
        if int(y) >= ytick[i] and int(y)<=ytick[i+1]:
            return r
        r+=1
    assert r<len(ytick), (y,r)
    return r

with codecs.open('msg_test_states.utf8', 'r') as ft:
    with codecs.open('msg_test_states_6k_gold.utf8', 'w') as fg:
        tlines = ft.readlines()
        for line in tlines:
            fg.write(STATES[getYClass(line.strip())])
            fg.write('\n')

print('FIN')