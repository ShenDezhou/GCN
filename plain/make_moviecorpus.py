import codecs
from sklearn.model_selection import ShuffleSplit

with codecs.open('movie_years.utf8', 'r', encoding='utf8') as fy:
    with codecs.open('movie_sentiments.utf8', 'r', encoding='utf8') as fs:
        with codecs.open('movie_actornames.utf8', 'r', encoding='utf8') as fa:
            with codecs.open('movie_states.utf8', 'r', encoding='utf8') as fm:
                with codecs.open('msg_training.utf8', 'w', encoding='utf8') as ft:
                    with codecs.open('msg_training_states.utf8', 'w', encoding='utf8') as fts:
                        with codecs.open('msg_test.utf8', 'w', encoding='utf8') as fte:
                            with codecs.open('msg_test_states.utf8', 'w', encoding='utf8') as ftes:
                                ylines = fy.readlines()
                                senlines = fs.readlines()
                                alines = fa.readlines()
                                slines = fm.readlines()
                                assert len(ylines) == len(senlines) and len(senlines)==len(alines) and len(alines)==len(slines)
                                rs = ShuffleSplit(n_splits=5,train_size=0.8, test_size=0.2, random_state=0)
                                for tr, te in rs.split(ylines):
                                    assert len(tr)==1036
                                    for i in tr:
                                        ft.write(ylines[i].strip())
                                        ft.write(',')
                                        segs = senlines[i].strip().split(',')
                                        ft.write(','.join(segs))
                                        ft.write(',')
                                        ft.write(str(int(segs[0])+int(segs[1])))
                                        ft.write(',')
                                        pnr = str(float(segs[0]) / float(segs[1])) if segs[1]!='0' else '0'
                                        ft.write(pnr)
                                        ft.write(',')
                                        ft.write(alines[i])

                                        fts.write(slines[i].strip())
                                        fts.write('\n')


                                    for i in te:
                                        fte.write(ylines[i].strip())
                                        fte.write(',')
                                        segs = senlines[i].strip().split(',')
                                        fte.write(','.join(segs))
                                        fte.write(',')
                                        fte.write(str(int(segs[0]) + int(segs[1])))
                                        fte.write(',')
                                        pnr = str(float(segs[0]) / float(segs[1])) if segs[1] != '0' else '0'
                                        fte.write(pnr)
                                        fte.write(',')
                                        fte.write(alines[i])

                                        ftes.write(slines[i])
                                    break

with codecs.open('../weibo_dic/movie_ndic.utf8', 'r', encoding='utf8') as fd:
    with codecs.open('moviename_training.utf8', 'w', encoding='utf8') as ftr:
        with codecs.open('moviename_test.utf8', 'w', encoding='utf8') as fte:
            dlines = fd.readlines()
            for i in tr:
                ftr.write(dlines[i].strip())
                ftr.write('\n')
            for i in te:
                fte.write(dlines[i].strip())
                fte.write('\n')

print('FIN')
