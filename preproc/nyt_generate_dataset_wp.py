import sys
sys.path.insert(0, '.')
from event_tensors.train_utils import RandomizedQueuedInstances
from event_tensors.glove_utils import Glove


if __name__ == '__main__':
    svo_file = 'data/svo_small.txt'
    output_file = 'data/word_prediction_small.txt'

    emb_file = 'data/glove.6B.100d.ext.txt'
    num_queues = 256
    batch_size = 128
    max_phrase_size = 10
    embeddings = Glove(emb_file)
    id2word = embeddings.reverse_dict()
    # remove None at the last
    data = list(iter(RandomizedQueuedInstances(svo_file, embeddings, num_queues, batch_size, max_phrase_size)))[:-1]

    output_file = open(output_file, 'w')
    for subj, verb, obj, word_id in data:
        subj_id, _ = subj
        verb_id, _ = verb
        obj_id, _ = obj
        subj = [id2word[i] for i in subj_id if i != 1]
        verb = [id2word[i] for i in verb_id if i != 1]
        obj = [id2word[i] for i in obj_id if i != 1]
        word = id2word[word_id]
        if len(subj_id) == 0:
            continue
        if len(verb_id) == 0:
            continue
        if len(obj_id) == 0:
            continue
        subj = ' '.join(subj)
        verb = ' '.join(verb)
        obj = ' '.join(obj)
        line = ' | '.join([subj, verb, obj, word]) + '\n'
        output_file.write(line)
    output_file.close()
