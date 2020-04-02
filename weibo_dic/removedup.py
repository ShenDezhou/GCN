import codecs

with codecs.open('movie_dic.utf8', 'r', encoding='utf8') as fa:
    with codecs.open('movie_ndic.utf8', 'w', encoding='utf8') as fb:
        lines = fa.readlines()
        lines = [l.strip() for l in lines]
        uniq = list(set(lines))
        print(len(uniq))
        uniq.sort()
        for movie in uniq:
            fb.write(movie+'\n')
print('FIN')
#1296