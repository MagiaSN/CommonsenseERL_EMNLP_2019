import sys


if __name__ == '__main__':
    sentic_path = '/users4/kliao/data/senticnet-5.0'
    input_file = '/users4/kliao/working/data/atomic/test.txt'
    output_file = '/users4/kliao/working/data/atomic/test2.txt'

    POLARITY_VALUE = 7

    sys.path.insert(0, sentic_path)
    from senticnet5 import senticnet
    senti_dict = { key: float(senticnet[key][POLARITY_VALUE]) for key in senticnet }

    lines = open(input_file, 'r').readlines()
    output_file = open(output_file, 'w')
    for line in lines:
        line = line.strip().split(' | ')
        subj, verb, obj, neg_obj, intent, neg_intent, react, neg_react = line
        words = [word for word in react.split(' ') if word in senti_dict]
        senti = sum([senti_dict[word] for word in words])
        senti = 1 if senti > 0 else 0
        output_file.write(' | '.join([subj, verb, obj, intent, neg_intent, str(senti)]) + '\n')
    output_file.close()
