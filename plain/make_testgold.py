import codecs

ytick = [0, 263.5, 244001]
STATES = list("AB")
def getYClass(y):
    r = 0
    for i in range(len(ytick)-1):
        if int(y) >= ytick[i] and int(y)<=ytick[i+1]:
            return r
        r+=1
    assert r<len(ytick), (y,r)
    return r

with codecs.open('msg_test_states.utf8', 'r') as ft:
    with codecs.open('msg_test_states_gold.utf8', 'w') as fg:
        tlines = ft.readlines()
        for line in tlines:
            fg.write(STATES[getYClass(line.strip())])
            fg.write('\n')

print('FIN')