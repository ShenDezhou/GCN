import codecs
#225
#112
#85
with codecs.open('../plain/movie_actornames.utf8', 'r', encoding='utf8') as fa:
    lines = fa.readlines()
    lens = [len(l.split(',')) for l in lines]
    lens = [i for i in lens if i<85]
    print(lens)
    print(max(lens))