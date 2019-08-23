import sys
import random


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        svo_intent_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        svo_intent_file = 'data/atomic/train_svo.txt'
        output_file = 'data/atomic/train.txt'

    neg_intent_file = 'data/atomic/negative_intents.txt'
    neg_react_file = 'data/atomic/negative_reacts.txt'
    emb_file = 'data/glove.6B.100d.ext.txt'

    lines = open(svo_intent_file, 'r').readlines()
    data = [line.strip().split(' | ') for line in lines]

    intents = [line.strip() for line in open(neg_intent_file, 'r').readlines()]
    reacts = [line.strip() for line in open(neg_react_file, 'r').readlines()]
    words = [line.strip().split(' ')[0] for line in open(emb_file, 'r').readlines()]

    neg_intents = random.sample(intents, len(data))
    neg_reacts = random.sample(reacts, len(data))
    neg_words = random.sample(words, len(data))
    for i in range(len(data)):
        subj, verb, obj, intent, react = data[i]
        neg_obj = neg_words[i]
        neg_intent = neg_intents[i]
        neg_react = neg_reacts[i]
        if intent[-1] == '.':
            intent = intent[:-1]
        if neg_intent[-1] == '.':
            neg_intent = neg_intent[:-1]
        if react[-1] == '.':
            react = react[:-1]
        if neg_react[-1] == '.':
            neg_react = neg_react[:-1]
        data[i] = [subj, verb, obj, neg_obj, intent, neg_intent, react, neg_react]

    output_file = open(output_file, 'w')
    for instance in data:
        output_file.write(' | '.join(instance) + '\n')
    output_file.close()
