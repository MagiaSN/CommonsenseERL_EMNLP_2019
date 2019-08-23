# filter instances with non-none intents in atomic data
import pandas as pd
import sys


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = 'data/atomic/v4_atomic_trn.csv'
        output_file = 'data/atomic/train_intent_react.txt'

    converters = {
        'xIntent': eval,
        'xReact': eval
    }

    data_frame = pd.read_csv(input_file, converters=converters)
    event_col = data_frame['event']
    intents_col = data_frame['xIntent']
    reacts_col = data_frame['xReact']

    output_file = open(output_file, 'w')
    for event, intents, reacts in zip(event_col, intents_col, reacts_col):
        if len(intents) == 0:
            continue
        if len(reacts) == 0:
            continue
        if len(intents) == 1 and intents[0].lower() == 'none':
            continue
        if len(reacts) == 1 and reacts[0].lower() == 'none':
            continue
        for intent in intents:
            for react in reacts:
                output_file.write(' | '.join([event, intent, react]) + '\n')
    output_file.close()
