import torch
import torch.utils.data
import numpy as np
import itertools

# use 1 (word ",") as padding

class WordPredictionDataset(torch.utils.data.Dataset):
    '''
    (
        subj_id: (max_phrase_size),
        subj_w:  (max_phrase_size),
        verb_id: (max_phrase_size),
        verb_w:  (max_phrase_size),
        obj_id:  (max_phrase_size),
        obj_w:   (max_phrase_size),
        word_id: int
    )
    '''
    def __init__(self):
        super(WordPredictionDataset, self).__init__()
        self.data = []

    def load(self, filename, embeddings, max_phrase_size=10):
        lines = open(filename, 'r').readlines()
        for line in lines:
            subj, verb, obj, word = line.strip().split(' | ')
            subj_id, subj_w = embeddings.transform(subj.split(' '), max_phrase_size)
            verb_id, verb_w = embeddings.transform(verb.split(' '), max_phrase_size)
            obj_id, obj_w = embeddings.transform(obj.split(' '), max_phrase_size)
            if subj_id is None or verb_id is None or obj_id is None or word not in embeddings.vocab:
                continue
            word_id = embeddings.id(word)
            self.data.append((subj_id, subj_w, verb_id, verb_w, obj_id, obj_w, word_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def WordPredictionDataset_collate_fn(samples):
    '''
    (
        subj_id: (batch, max_phrase_size),
        subj_w:  (batch, max_phrase_size),
        verb_id: (batch, max_phrase_size),
        verb_w:  (batch, max_phrase_size),
        obj_id:  (batch, max_phrase_size),
        obj_w:   (batch, max_phrase_size),
        word_id: (batch)
    )
    '''
    subj_id = torch.LongTensor([sample[0] for sample in samples])
    subj_w = torch.FloatTensor([sample[1] for sample in samples])
    verb_id = torch.LongTensor([sample[2] for sample in samples])
    verb_w = torch.FloatTensor([sample[3] for sample in samples])
    obj_id = torch.LongTensor([sample[4] for sample in samples])
    obj_w = torch.FloatTensor([sample[5] for sample in samples])
    word_id = torch.LongTensor([sample[6] for sample in samples])
    return subj_id, subj_w, verb_id, verb_w, obj_id, obj_w, word_id


class EventPredictionDataset(torch.utils.data.Dataset):
    '''
    (
        (
            (
                ei_subj_id: (max_phrase_size),
                ei_subj_w:  (max_phrase_size)
            ),
            (
                ei_verb_id: (max_phrase_size),
                ei_verb_w:  (max_phrase_size)
            ),
            (
                ei_obj_id: (max_phrase_size),
                ei_obj_w:  (max_phrase_size)
            )
        ),
        (
            (
                et_subj_id: (max_phrase_id),
                et_subj_w:  (max_phrase_id)
            ),
            (
                et_verb_id: (max_phrase_id),
                et_verb_w:  (max_phrase_id)
            ),
            (
                et_obj_id: (max_phrase_id),
                et_obj_w:  (max_phrase_id)
            )
        ),
        (
            (
                en_subj_id: (max_phrase_id),
                en_subj_w:  (max_phrase_id)
            ),
            (
                en_verb_id: (max_phrase_id),
                en_verb_w:  (max_phrase_id)
            ),
            (
                en_obj_id: (max_phrase_id),
                en_obj_w:  (max_phrase_id)
            )
        )
    )
    '''
    def __init__(self):
        super(EventPredictionDataset, self).__init__()
        self.data = []

    def load(self, filename, embeddings, max_phrase_size=10):
        self.data = []
        f = open(filename, 'r')
        for line in f:
            line = line.strip()
            ei, et, en = line.split(', ')
            ei_subj, ei_verb, ei_obj = ei.split('|')
            et_subj, et_verb, et_obj = et.split('|')
            en_subj, en_verb, en_obj = en.split('|')
            ei_subj_id, ei_subj_w = embeddings.transform(ei_subj.split(' '), max_phrase_size)
            ei_verb_id, ei_verb_w = embeddings.transform(ei_verb.split(' '), max_phrase_size)
            ei_obj_id, ei_obj_w = embeddings.transform(ei_obj.split(' '), max_phrase_size)
            et_subj_id, et_subj_w = embeddings.transform(et_subj.split(' '), max_phrase_size)
            et_verb_id, et_verb_w = embeddings.transform(et_verb.split(' '), max_phrase_size)
            et_obj_id, et_obj_w = embeddings.transform(et_obj.split(' '), max_phrase_size)
            en_subj_id, en_subj_w = embeddings.transform(en_subj.split(' '), max_phrase_size)
            en_verb_id, en_verb_w = embeddings.transform(en_verb.split(' '), max_phrase_size)
            en_obj_id, en_obj_w = embeddings.transform(en_obj.split(' '), max_phrase_size)
            # if ei_subj_id is None or ei_verb_id is None or ei_obj_id is None:
            #     print(ei)
            # if et_subj_id is None or et_verb_id is None or et_obj_id is None:
            #     print(et)
            # if en_subj_id is None or en_verb_id is None or en_obj_id is None:
            #     print(en)
            instance = (
                ((ei_subj_id, ei_subj_w), (ei_verb_id, ei_verb_w), (ei_obj_id, ei_obj_w)),
                ((et_subj_id, et_subj_w), (et_verb_id, et_verb_w), (et_obj_id, et_obj_w)),
                ((en_subj_id, en_subj_w), (en_verb_id, en_verb_w), (en_obj_id, en_obj_w))
            )
            self.data.append(instance)
        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def EventPredictionDataset_collate_fn(samples):
    '''
    (
        ei_subj_id: (batch, max_phrase_size),
        ei_subj_w:  (batch, max_phrase_size),
        ei_verb_id: (batch, max_phrase_size),
        ei_verb_w:  (batch, max_phrase_size),
        ei_obj_id:  (batch, max_phrase_size),
        ei_obj_w:   (batch, max_phrase_size),

        et_subj_id: (batch, max_phrase_size),
        et_subj_w:  (batch, max_phrase_size),
        et_verb_id: (batch, max_phrase_size),
        et_verb_w:  (batch, max_phrase_size),
        et_obj_id:  (batch, max_phrase_size),
        et_obj_w:   (batch, max_phrase_size),

        en_subj_id: (batch, max_phrase_size),
        en_subj_w:  (batch, max_phrase_size),
        en_verb_id: (batch, max_phrase_size),
        en_verb_w:  (batch, max_phrase_size),
        en_obj_id:  (batch, max_phrase_size),
        en_obj_w:   (batch, max_phrase_size)
    )
    '''
    ei_subj_id = []
    ei_subj_w = []
    ei_verb_id = []
    ei_verb_w = []
    ei_obj_id = []
    ei_obj_w = []
    et_subj_id = []
    et_subj_w = []
    et_verb_id = []
    et_verb_w = []
    et_obj_id = []
    et_obj_w = []
    en_subj_id = []
    en_subj_w = []
    en_verb_id = []
    en_verb_w = []
    en_obj_id = []
    en_obj_w = []
    for sample in samples:
        ei_subj_id.append(sample[0][0][0])
        ei_subj_w.append(sample[0][0][1])
        ei_verb_id.append(sample[0][1][0])
        ei_verb_w.append(sample[0][1][1])
        ei_obj_id.append(sample[0][2][0])
        ei_obj_w.append(sample[0][2][1])
        et_subj_id.append(sample[1][0][0])
        et_subj_w.append(sample[1][0][1])
        et_verb_id.append(sample[1][1][0])
        et_verb_w.append(sample[1][1][1])
        et_obj_id.append(sample[1][2][0])
        et_obj_w.append(sample[1][2][1])
        en_subj_id.append(sample[2][0][0])
        en_subj_w.append(sample[2][0][1])
        en_verb_id.append(sample[2][1][0])
        en_verb_w.append(sample[2][1][1])
        en_obj_id.append(sample[2][2][0])
        en_obj_w.append(sample[2][2][1])
    ei_subj_id = torch.from_numpy(np.array(ei_subj_id)).long()
    ei_subj_w = torch.from_numpy(np.array(ei_subj_w)).float()
    ei_verb_id = torch.from_numpy(np.array(ei_verb_id)).long()
    ei_verb_w = torch.from_numpy(np.array(ei_verb_w)).float()
    ei_obj_id = torch.from_numpy(np.array(ei_obj_id)).long()
    ei_obj_w = torch.from_numpy(np.array(ei_obj_w)).float()
    et_subj_id = torch.from_numpy(np.array(et_subj_id)).long()
    et_subj_w = torch.from_numpy(np.array(et_subj_w)).float()
    et_verb_id = torch.from_numpy(np.array(et_verb_id)).long()
    et_verb_w = torch.from_numpy(np.array(et_verb_w)).float()
    et_obj_id = torch.from_numpy(np.array(et_obj_id)).long()
    et_obj_w = torch.from_numpy(np.array(et_obj_w)).float()
    en_subj_id = torch.from_numpy(np.array(en_subj_id)).long()
    en_subj_w = torch.from_numpy(np.array(en_subj_w)).float()
    en_verb_id = torch.from_numpy(np.array(en_verb_id)).long()
    en_verb_w = torch.from_numpy(np.array(en_verb_w)).float()
    en_obj_id = torch.from_numpy(np.array(en_obj_id)).long()
    en_obj_w = torch.from_numpy(np.array(en_obj_w)).float()
    return ei_subj_id, ei_subj_w, ei_verb_id, ei_verb_w, ei_obj_id, ei_obj_w, \
           et_subj_id, et_subj_w, et_verb_id, et_verb_w, et_obj_id, et_obj_w, \
           en_subj_id, en_subj_w, en_verb_id, en_verb_w, en_obj_id, en_obj_w


class EventIntentSentimentDataset(torch.utils.data.Dataset):
    '''
    (
        subj_id:       (max_phrase_size),
        subj_w:        (max_phrase_size),
        verb_id:       (max_phrase_size),
        verb_w:        (max_phrase_size),
        obj_id:        (max_phrase_size),
        obj_w:         (max_phrase_size),
        neg_obj_id:    (max_phrase_size),
        neg_obj_w:     (max_phrase_size),
        intent_id:     (max_phrase_size),
        neg_intent_id: (max_phrase_size),
        sentiment:     int
        neg_sentiment: int
    )
    '''
    def __init__(self):
        super(EventIntentSentimentDataset, self).__init__()
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def load(self, filename, embeddings, max_phrase_size=10):
        self.data = []
        f = open(filename, 'r')
        for line in f:
            subj, verb, obj, neg_obj, intent, neg_intent, sentiment = line.strip().lower().split(' | ')
            subj_id, subj_w = embeddings.transform(subj.split(' '), max_phrase_size)
            verb_id, verb_w = embeddings.transform(verb.split(' '), max_phrase_size)
            obj_id, obj_w = embeddings.transform(obj.split(' '), max_phrase_size)
            neg_obj_id, neg_obj_w = embeddings.transform(neg_obj.split(' '), max_phrase_size)
            intent_id, _ = embeddings.transform(intent.split(' '), max_phrase_size)
            neg_intent_id, _ = embeddings.transform(neg_intent.split(' '), max_phrase_size)
            sentiment = int(sentiment)
            neg_sentiment = 1 - sentiment
            if intent_id is None or neg_intent_id is None:
                continue
            if subj_id is None:
                subj_id = np.ones(max_phrase_size)
                subj_w = np.zeros(max_phrase_size)
            if verb_id is None:
                verb_id = np.ones(max_phrase_size)
                verb_w = np.zeros(max_phrase_size)
            if obj_id is None:
                obj_id = np.ones(max_phrase_size)
                obj_w = np.zeros(max_phrase_size)
            if neg_obj_id is None:
                neg_obj_id = np.ones(max_phrase_size)
                neg_obj_w = np.zeros(max_phrase_size)
            self.data.append((
                subj_id, subj_w, verb_id, verb_w, obj_id, obj_w, neg_obj_id, neg_obj_w,
                intent_id, neg_intent_id, sentiment, neg_sentiment
            ))
        f.close()

def EventIntentSentimentDataset_collate_fn(samples):
    '''
    (
        subj_id:       (batch, max_phrase_size),
        subj_w:        (batch, max_phrase_size),
        verb_id:       (batch, max_phrase_size),
        verb_w:        (batch, max_phrase_size),
        obj_id:        (batch, max_phrase_size),
        obj_w:         (batch, max_phrase_size),
        neg_obj_id:    (batch, max_phrase_size),
        neg_obj_w:     (batch, max_phrase_size),
        intent_id:     (batch, max_phrase_size),
        neg_intent_id: (batch, max_phrase_size),
        sentiment:     (batch),
        neg_sentiment: (batch)
    )
    '''
    subj_id = []
    subj_w = []
    verb_id = []
    verb_w = []
    obj_id = []
    obj_w = []
    neg_obj_id = []
    neg_obj_w = []
    intent_id = []
    neg_intent_id = []
    sentiment = []
    neg_sentiment = []
    for sample in samples:
        subj_id.append(sample[0])
        subj_w.append(sample[1])
        verb_id.append(sample[2])
        verb_w.append(sample[3])
        obj_id.append(sample[4])
        obj_w.append(sample[5])
        neg_obj_id.append(sample[6])
        neg_obj_w.append(sample[7])
        intent_id.append(sample[8])
        neg_intent_id.append(sample[9])
        sentiment.append(sample[10])
        neg_sentiment.append(sample[11])
    subj_id = torch.from_numpy(np.array(subj_id)).long()
    subj_w = torch.from_numpy(np.array(subj_w)).float()
    verb_id = torch.from_numpy(np.array(verb_id)).long()
    verb_w = torch.from_numpy(np.array(verb_w)).float()
    obj_id = torch.from_numpy(np.array(obj_id)).long()
    obj_w = torch.from_numpy(np.array(obj_w)).float()
    neg_obj_id = torch.from_numpy(np.array(neg_obj_id)).long()
    neg_obj_w = torch.from_numpy(np.array(neg_obj_w)).float()
    intent_id = torch.from_numpy(np.array(intent_id)).long()
    neg_intent_id = torch.from_numpy(np.array(neg_intent_id)).long()
    sentiment = torch.from_numpy(np.array(sentiment)).long()
    neg_sentiment = torch.from_numpy(np.array(neg_sentiment)).long()
    return subj_id, subj_w, verb_id, verb_w, obj_id, obj_w, neg_obj_id, neg_obj_w, \
           intent_id, neg_intent_id, sentiment, neg_sentiment


class HardSimilarityDataset(torch.utils.data.Dataset):
    '''
    (
        pos_e1_subj_id: (max_phrase_size),
        pos_e1_subj_w:  (max_phrase_size),
        pos_e1_verb_id: (max_phrase_size),
        pos_e1_verb_w:  (max_phrase_size),
        pos_e1_obj_id:  (max_phrase_size),
        pos_e1_obj_w:   (max_phrase_size),
        pos_e2_subj_id: (max_phrase_size),
        pos_e2_subj_w:  (max_phrase_size),
        pos_e2_verb_id: (max_phrase_size),
        pos_e2_verb_w:  (max_phrase_size),
        pos_e2_obj_id:  (max_phrase_size),
        pos_e2_obj_w:   (max_phrase_size),
        neg_e1_subj_id: (max_phrase_size),
        neg_e1_subj_w:  (max_phrase_size),
        neg_e1_verb_id: (max_phrase_size),
        neg_e1_verb_w:  (max_phrase_size),
        neg_e1_obj_id:  (max_phrase_size),
        neg_e1_obj_w:   (max_phrase_size),
        neg_e2_subj_id: (max_phrase_size),
        neg_e2_subj_w:  (max_phrase_size),
        neg_e2_verb_id: (max_phrase_size),
        neg_e2_verb_w:  (max_phrase_size),
        neg_e2_obj_id:  (max_phrase_size),
        neg_e2_obj_w:   (max_phrase_size)
    )
    '''
    def __init__(self, file, embeddings, max_phrase_size=10):
        super(HardSimilarityDataset, self).__init__()
        self.data = []

        lines = open(file, 'r').readlines()
        for line in lines:
            pos_e1_subj, pos_e1_verb, pos_e1_obj, pos_e2_subj, pos_e2_verb, pos_e2_obj, \
            neg_e1_subj, neg_e1_verb, neg_e1_obj, neg_e2_subj, neg_e2_verb, neg_e2_obj = line.strip().split(' | ')
            pos_e1_subj = pos_e1_subj.split(' ')
            pos_e1_verb = pos_e1_verb.split(' ')
            pos_e1_obj = pos_e1_obj.split(' ')
            pos_e2_subj = pos_e2_subj.split(' ')
            pos_e2_verb = pos_e2_verb.split(' ')
            pos_e2_obj = pos_e2_obj.split(' ')
            neg_e1_subj = neg_e1_subj.split(' ')
            neg_e1_verb = neg_e1_verb.split(' ')
            neg_e1_obj = neg_e1_obj.split(' ')
            neg_e2_subj = neg_e2_subj.split(' ')
            neg_e2_verb = neg_e2_verb.split(' ')
            neg_e2_obj = neg_e2_obj.split(' ')
            pos_e1_subj_id, pos_e1_subj_w = embeddings.transform(pos_e1_subj, max_phrase_size)
            pos_e1_verb_id, pos_e1_verb_w = embeddings.transform(pos_e1_verb, max_phrase_size)
            pos_e1_obj_id, pos_e1_obj_w = embeddings.transform(pos_e1_obj, max_phrase_size)
            pos_e2_subj_id, pos_e2_subj_w = embeddings.transform(pos_e2_subj, max_phrase_size)
            pos_e2_verb_id, pos_e2_verb_w = embeddings.transform(pos_e2_verb, max_phrase_size)
            pos_e2_obj_id, pos_e2_obj_w = embeddings.transform(pos_e2_obj, max_phrase_size)
            neg_e1_subj_id, neg_e1_subj_w = embeddings.transform(neg_e1_subj, max_phrase_size)
            neg_e1_verb_id, neg_e1_verb_w = embeddings.transform(neg_e1_verb, max_phrase_size)
            neg_e1_obj_id, neg_e1_obj_w = embeddings.transform(neg_e1_obj, max_phrase_size)
            neg_e2_subj_id, neg_e2_subj_w = embeddings.transform(neg_e2_subj, max_phrase_size)
            neg_e2_verb_id, neg_e2_verb_w = embeddings.transform(neg_e2_verb, max_phrase_size)
            neg_e2_obj_id, neg_e2_obj_w = embeddings.transform(neg_e2_obj, max_phrase_size)
            self.data.append((
                pos_e1_subj_id, pos_e1_subj_w, pos_e1_verb_id, pos_e1_verb_w, pos_e1_obj_id, pos_e1_obj_w,
                pos_e2_subj_id, pos_e2_subj_w, pos_e2_verb_id, pos_e2_verb_w, pos_e2_obj_id, pos_e2_obj_w,
                neg_e1_subj_id, neg_e1_subj_w, neg_e1_verb_id, neg_e1_verb_w, neg_e1_obj_id, neg_e1_obj_w,
                neg_e2_subj_id, neg_e2_subj_w, neg_e2_verb_id, neg_e2_verb_w, neg_e2_obj_id, neg_e2_obj_w
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def HardSimilarityDataset_collate_fn(samples):
    '''
    (
        pos_e1_subj_id: (batch, n),
        pos_e1_subj_w:  (batch, n),
        pos_e1_verb_id: (batch, n),
        pos_e1_verb_w:  (batch, n),
        pos_e1_obj_id:  (batch, n),
        pos_e1_obj_w:   (batch, n),
        pos_e2_subj_id: (batch, n),
        pos_e2_subj_w:  (batch, n),
        pos_e2_verb_id: (batch, n),
        pos_e2_verb_w:  (batch, n),
        pos_e2_obj_id:  (batch, n),
        pos_e2_obj_w:   (batch, n),
        neg_e1_subj_id: (batch, n),
        neg_e1_subj_w:  (batch, n),
        neg_e1_verb_id: (batch, n),
        neg_e1_verb_w:  (batch, n),
        neg_e1_obj_id:  (batch, n),
        neg_e1_obj_w:   (batch, n),
        neg_e2_subj_id: (batch, n),
        neg_e2_subj_w:  (batch, n),
        neg_e2_verb_id: (batch, n),
        neg_e2_verb_w:  (batch, n),
        neg_e2_obj_id:  (batch, n),
        neg_e2_obj_w:   (batch, n)
    )
    '''
    pos_e1_subj_id = torch.LongTensor([sample[0] for sample in samples])
    pos_e1_subj_w = torch.FloatTensor([sample[1] for sample in samples])
    pos_e1_verb_id = torch.LongTensor([sample[2] for sample in samples])
    pos_e1_verb_w = torch.FloatTensor([sample[3] for sample in samples])
    pos_e1_obj_id = torch.LongTensor([sample[4] for sample in samples])
    pos_e1_obj_w = torch.FloatTensor([sample[5] for sample in samples])
    pos_e2_subj_id = torch.LongTensor([sample[6] for sample in samples])
    pos_e2_subj_w = torch.FloatTensor([sample[7] for sample in samples])
    pos_e2_verb_id = torch.LongTensor([sample[8] for sample in samples])
    pos_e2_verb_w = torch.FloatTensor([sample[9] for sample in samples])
    pos_e2_obj_id = torch.LongTensor([sample[10] for sample in samples])
    pos_e2_obj_w = torch.FloatTensor([sample[11] for sample in samples])
    neg_e1_subj_id = torch.LongTensor([sample[12] for sample in samples])
    neg_e1_subj_w = torch.FloatTensor([sample[13] for sample in samples])
    neg_e1_verb_id = torch.LongTensor([sample[14] for sample in samples])
    neg_e1_verb_w = torch.FloatTensor([sample[15] for sample in samples])
    neg_e1_obj_id = torch.LongTensor([sample[16] for sample in samples])
    neg_e1_obj_w = torch.FloatTensor([sample[17] for sample in samples])
    neg_e2_subj_id = torch.LongTensor([sample[18] for sample in samples])
    neg_e2_subj_w = torch.FloatTensor([sample[19] for sample in samples])
    neg_e2_verb_id = torch.LongTensor([sample[20] for sample in samples])
    neg_e2_verb_w = torch.FloatTensor([sample[21] for sample in samples])
    neg_e2_obj_id = torch.LongTensor([sample[22] for sample in samples])
    neg_e2_obj_w = torch.FloatTensor([sample[23] for sample in samples])
    return pos_e1_subj_id, pos_e1_subj_w, pos_e1_verb_id, pos_e1_verb_w, pos_e1_obj_id, pos_e1_obj_w, \
           pos_e2_subj_id, pos_e2_subj_w, pos_e2_verb_id, pos_e2_verb_w, pos_e2_obj_id, pos_e2_obj_w, \
           neg_e1_subj_id, neg_e1_subj_w, neg_e1_verb_id, neg_e1_verb_w, neg_e1_obj_id, neg_e1_obj_w, \
           neg_e2_subj_id, neg_e2_subj_w, neg_e2_verb_id, neg_e2_verb_w, neg_e2_obj_id, neg_e2_obj_w


class TransitiveSentenceSimilarityDataset(torch.utils.data.Dataset):
    '''
    (
        e1_subj_id: (max_phrase_size),
        e1_subj_w:  (max_phrase_size),
        e1_verb_id: (max_phrase_size),
        e1_verb_w:  (max_phrase_size),
        e1_obj_id:  (max_phrase_size),
        e1_obj_w:   (max_phrase_size),
        e2_subj_id: (max_phrase_size),
        e2_subj_w:  (max_phrase_size),
        e2_verb_id: (max_phrase_size),
        e2_verb_w:  (max_phrase_size),
        e2_obj_id:  (max_phrase_size),
        e2_obj_w:   (max_phrase_size),
        score:      float
    )
    '''
    def __init__(self, file, embeddings, max_phrase_size=10):
        super(TransitiveSentenceSimilarityDataset, self).__init__()
        self.data = []

        lines = open(file, 'r').readlines()
        for line in lines:
            e1_subj, e1_verb, e1_obj, e2_subj, e2_verb, e2_obj, score = line.strip().split(' | ')
            e1_subj = e1_subj.split(' ')
            e1_verb = e1_verb.split(' ')
            e1_obj = e1_obj.split(' ')
            e2_subj = e2_subj.split(' ')
            e2_verb = e2_verb.split(' ')
            e2_obj = e2_obj.split(' ')
            score = float(score)
            e1_subj_id, e1_subj_w = embeddings.transform(e1_subj, max_phrase_size)
            e1_verb_id, e1_verb_w = embeddings.transform(e1_verb, max_phrase_size)
            e1_obj_id, e1_obj_w = embeddings.transform(e1_obj, max_phrase_size)
            e2_subj_id, e2_subj_w = embeddings.transform(e2_subj, max_phrase_size)
            e2_verb_id, e2_verb_w = embeddings.transform(e2_verb, max_phrase_size)
            e2_obj_id, e2_obj_w = embeddings.transform(e2_obj, max_phrase_size)
            self.data.append((
                e1_subj_id, e1_subj_w, e1_verb_id, e1_verb_w, e1_obj_id, e1_obj_w,
                e2_subj_id, e2_subj_w, e2_verb_id, e2_verb_w, e2_obj_id, e2_obj_w,
                score
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def TransitiveSentenceSimilarityDataset_collate_fn(samples):
    '''
    (
        e1_subj_id: (batch, n),
        e1_subj_w:  (batch, n),
        e1_verb_id: (batch, n),
        e1_verb_w:  (batch, n),
        e1_obj_id:  (batch, n),
        e1_obj_w:   (batch, n),
        e2_subj_id: (batch, n),
        e2_subj_w:  (batch, n),
        e2_verb_id: (batch, n),
        e2_verb_w:  (batch, n),
        e2_obj_id:  (batch, n),
        e2_obj_w:   (batch, n),
        score:      (batch)
    )
    '''
    e1_subj_id = torch.LongTensor([sample[0] for sample in samples])
    e1_subj_w = torch.FloatTensor([sample[1] for sample in samples])
    e1_verb_id = torch.LongTensor([sample[2] for sample in samples])
    e1_verb_w = torch.FloatTensor([sample[3] for sample in samples])
    e1_obj_id = torch.LongTensor([sample[4] for sample in samples])
    e1_obj_w = torch.FloatTensor([sample[5] for sample in samples])
    e2_subj_id = torch.LongTensor([sample[6] for sample in samples])
    e2_subj_w = torch.FloatTensor([sample[7] for sample in samples])
    e2_verb_id = torch.LongTensor([sample[8] for sample in samples])
    e2_verb_w = torch.FloatTensor([sample[9] for sample in samples])
    e2_obj_id = torch.LongTensor([sample[10] for sample in samples])
    e2_obj_w = torch.FloatTensor([sample[11] for sample in samples])
    scores = torch.FloatTensor([sample[12] for sample in samples])
    return e1_subj_id, e1_subj_w, e1_verb_id, e1_verb_w, e1_obj_id, e1_obj_w, \
           e2_subj_id, e2_subj_w, e2_verb_id, e2_verb_w, e2_obj_id, e2_obj_w, \
           scores
