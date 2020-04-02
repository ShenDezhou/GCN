import codecs

with codecs.open('../weibo_dic/cbooo_people.utf8', 'r', encoding='utf8') as fd:
    with codecs.open('../plain/movie_actors.utf8', 'r', encoding='utf8') as fa:
        with codecs.open('../plain/movie_actornames.utf8', 'w', encoding='utf8') as fs:
            with codecs.open('../plain/actor_dic.utf8', 'w', encoding='utf8') as fz:
                dlines = fd.readlines()
                rxdict = {}
                for line in dlines:
                    l = line.strip().split(',')
                    rxdict[l[-1]] = l[1]
                actors = fa.readlines()
                actors_dic = []
                for actor in actors:
                    actorids = actor.strip().split(',')
                    actornames = [rxdict[id] for id in actorids if len(rxdict[id])>0]
                    actors_dic.extend(actornames)
                    fs.write(",".join(actornames)+'\n')
                actors_dic = list(set(actors_dic))
                actors_dic.sort()
                for actorx in actors_dic:
                    fz.write(actorx+'\n')

print('FIN')



