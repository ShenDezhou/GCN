import codecs

with codecs.open('movie_sentiment.utf8', 'r', encoding='utf8') as fs:
    with codecs.open('../weibo_dic/movie_ndic.utf8', 'r', encoding='utf8') as fn:
        with codecs.open('../weibo_dic/movie_dic_raw.utf8', 'r', encoding='utf8') as fd:
            with codecs.open('../plain/movie_sentiments.utf8', 'w', encoding='utf8') as fo:
                xlines = fn.readlines()
                dlines = fd.readlines()
                dlines = [l.strip() for l in dlines]
                rxdict = dict(zip(dlines,range(1, 1+len(dlines))))
                senlines = fs.readlines()
                rsdict ={}
                for sline in senlines:
                    segs = sline.split(',', maxsplit=1)
                    assert len(segs) == 2
                    rsdict[segs[0]] = segs[1]
                print(rxdict)
                print(rsdict)
                for xline in xlines:
                    index = rxdict[xline.strip()]
                    sent = rsdict.get(str(index),"0,0")
                    fo.write(sent.strip()+'\n')
print('FIN')


