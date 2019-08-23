import random


if __name__ == '__main__':
    filenames = [
        'data/nyt_ollie/1987.txt',
        'data/nyt_ollie/1988.txt',
        'data/nyt_ollie/1989.txt',
        'data/nyt_ollie/1990.txt',
        'data/nyt_ollie/1991.txt',
        'data/nyt_ollie/1992.txt',
        'data/nyt_ollie/1993.txt',
        'data/nyt_ollie/1994.txt',
        'data/nyt_ollie/1995.txt',
        'data/nyt_ollie/1996.txt',
        'data/nyt_ollie/1997.txt',
        'data/nyt_ollie/1998.txt',
        'data/nyt_ollie/1999.txt',
        'data/nyt_ollie/2000.txt',
        'data/nyt_ollie/2001.txt',
        'data/nyt_ollie/2002.txt',
        'data/nyt_ollie/2003.txt',
        'data/nyt_ollie/2004.txt',
        'data/nyt_ollie/2005.txt',
        'data/nyt_ollie/2006.txt',
        'data/nyt_ollie/2007.txt',
    ]

    instances = []
    for filename in filenames:
        instances += open(filename, 'r').readlines()
    num_total = len(instances)

    def generate(num, output_file):
        indices = random.sample(range(num_total), num)
        samples = [instances[index] for index in indices]
        f = open(output_file, 'w')
        for line in samples:
            f.write(line)
        f.close()
        print(output_file + ' done')

    num_dict = {
        'data/nyt_final/1987_neg.txt': 6423165,
        'data/nyt_final/1988_neg.txt': 6491698,
        'data/nyt_final/1989_neg.txt': 6347525,
        'data/nyt_final/1990_neg.txt': 6243159,
        'data/nyt_final/1991_neg.txt': 5559770,
        'data/nyt_final/1992_neg.txt': 5447308,
        'data/nyt_final/1993_neg.txt': 5324032,
        'data/nyt_final/1994_neg.txt': 5288859,
        'data/nyt_final/1995_neg.txt': 5744247,
        'data/nyt_final/1996_neg.txt': 5772160,
        'data/nyt_final/1997_neg.txt': 5808079,
        'data/nyt_final/1998_neg.txt': 6424286,
        'data/nyt_final/1999_neg.txt': 6594287,
        'data/nyt_final/2000_neg.txt': 6977080,
        'data/nyt_final/2001_neg.txt': 6864116,
        'data/nyt_final/2002_neg.txt': 6978644,
        'data/nyt_final/2003_neg.txt': 6837301,
        'data/nyt_final/2004_neg.txt': 6703959,
        'data/nyt_final/2005_neg.txt': 6614429,
        'data/nyt_final/2006_neg.txt': 6662519,
        'data/nyt_final/2007_neg.txt': 3106803,
    }

    for filename in num_dict:
        num = num_dict[filename]
        generate(num, filename)
