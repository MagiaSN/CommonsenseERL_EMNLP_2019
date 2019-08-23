# collect all intents and reacts
import random
import os


if __name__ == '__main__':
    random.seed(19950125)

    filenames = ['train_intent_react.txt', 'dev_intent_react.txt', 'test_intent_react.txt']

    all_intents = []
    all_reacts = []
    for filename in filenames:
        lines = open(os.path.join('data/atomic', filename), 'r').readlines()
        for line in lines:
            event, intent, react = line.strip().split(' | ')
            all_intents.append(intent)
            all_reacts.append(react)
    
    random.shuffle(all_intents)
    random.shuffle(all_reacts)

    with open('data/atomic/negative_intents.txt', 'w') as f:
        for intent in all_intents:
            f.write(intent + '\n')
    with open('data/atomic/negative_reacts.txt', 'w') as f:
        for react in all_reacts:
            f.write(react + '\n')
