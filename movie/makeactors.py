import codecs

with codecs.open('moviebox.csv', 'r', encoding='utf8') as fc:
    with codecs.open('../weibo_dic/movie_ndic.utf8', 'r', encoding='utf8') as fm:
        with codecs.open('../plain/movie_actors.utf8', 'w', encoding='utf8') as fa:
            with codecs.open('../plain/movie_years.utf8', 'w', encoding='utf8') as fy:
                with codecs.open('../plain/movie_states.utf8', 'w', encoding='utf8') as fs:
                    movies = fm.readlines()
                    xlines = fc.readlines()
                    movie_features = {}
                    for movie in movies:
                        features = [l for l in xlines if l.startswith(movie.strip())]
                        assert len(features)>0,(movie, len(features))
                        actorids = [l.split(',')[1] for l in features]
                        year = features[0].split(',')[2]
                        gross = features[0].split(',')[3].strip()
                        if len(gross) == 0:
                            gross = '0'
                        fa.write(','.join(actorids)+'\n')
                        fy.write(year+'\n')
                        fs.write(gross+'\n')
print('FIN')



