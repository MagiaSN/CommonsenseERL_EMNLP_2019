import torch.nn as nn
import torch.utils.data
import sys
import logging
import argparse
from model import LowRankNeuralTensorNetwork, NeuralTensorNetwork, RoleFactoredTensorModel, PredicateTensorModel, Averaging, NN, EMC
from dataset import HardSimilarityDataset, HardSimilarityDataset_collate_fn
from event_tensors.glove_utils import Glove


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--vocab_size', type=int, default=400000)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--emb_file', type=str, default='data/glove.6B.100d.ext.txt')
    parser.add_argument('--dataset_file', type=str, default='data/hard.txt')
    parser.add_argument('--model_file', type=str, default='model/nyt/ntn/NeuralTensorNetwork_2007.pt')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--model', type=str, default='NTN')
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--em_r', type=int, default=10)
    option = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    glove = Glove(option.emb_file)
    logging.info('Embeddings loaded')

    dataset = HardSimilarityDataset(option.dataset_file, glove)
    logging.info('Dataset loaded')

    embeddings = nn.Embedding(option.vocab_size, option.emb_dim, padding_idx=1)
    if option.model == 'NTN':
        model = NeuralTensorNetwork(embeddings, option.em_k)
    elif option.model == 'LowRankNTN':
        model = LowRankNeuralTensorNetwork(embeddings, option.em_k, option.em_r)
    elif option.model == 'RoleFactor':
        model = RoleFactoredTensorModel(embeddings, option.em_k)
    elif option.model == 'Predicate':
        model = PredicateTensorModel(embeddings)
    elif option.model == 'NN':
        model = NN(embeddings, 2 * option.em_k, option.em_k)
    elif option.model == 'EMC':
        model = EMC(embeddings, 2 * option.em_k, option.em_k)
    else:
        logging.info('Unknown model type: ' + option.model)
        exit(1)

    checkpoint = torch.load(option.model_file)
    if type(checkpoint) == dict:
        if 'event_model_state_dict' in checkpoint:
            state_dict = checkpoint['event_model_state_dict']
        else:
            state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    logging.info(option.model_file + ' loaded')

    # embeddings = nn.Embedding(option.vocab_size, option.emb_dim, padding_idx=1)
    # embeddings.weight.data = torch.from_numpy(glove.embd).float()
    # model = Averaging(embeddings)

    if option.use_gpu:
        model.cuda()
    model.eval()

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=HardSimilarityDataset_collate_fn, shuffle=False, batch_size=len(dataset))
    batch = next(iter(data_loader))
    pos_e1_subj_id, pos_e1_subj_w, pos_e1_verb_id, pos_e1_verb_w, pos_e1_obj_id, pos_e1_obj_w, \
    pos_e2_subj_id, pos_e2_subj_w, pos_e2_verb_id, pos_e2_verb_w, pos_e2_obj_id, pos_e2_obj_w, \
    neg_e1_subj_id, neg_e1_subj_w, neg_e1_verb_id, neg_e1_verb_w, neg_e1_obj_id, neg_e1_obj_w, \
    neg_e2_subj_id, neg_e2_subj_w, neg_e2_verb_id, neg_e2_verb_w, neg_e2_obj_id, neg_e2_obj_w = batch

    if option.use_gpu:
        pos_e1_subj_id = pos_e1_subj_id.cuda()
        pos_e1_subj_w = pos_e1_subj_w.cuda()
        pos_e1_verb_id = pos_e1_verb_id.cuda()
        pos_e1_verb_w = pos_e1_verb_w.cuda()
        pos_e1_obj_id = pos_e1_obj_id.cuda()
        pos_e1_obj_w = pos_e1_obj_w.cuda()
        pos_e2_subj_id = pos_e2_subj_id.cuda()
        pos_e2_subj_w = pos_e2_subj_w.cuda()
        pos_e2_verb_id = pos_e2_verb_id.cuda()
        pos_e2_verb_w = pos_e2_verb_w.cuda()
        pos_e2_obj_id = pos_e2_obj_id.cuda()
        pos_e2_obj_w = pos_e2_obj_w.cuda()
        neg_e1_subj_id = neg_e1_subj_id.cuda()
        neg_e1_subj_w = neg_e1_subj_w.cuda()
        neg_e1_verb_id = neg_e1_verb_id.cuda()
        neg_e1_verb_w = neg_e1_verb_w.cuda()
        neg_e1_obj_id = neg_e1_obj_id.cuda()
        neg_e1_obj_w = neg_e1_obj_w.cuda()
        neg_e2_subj_id = neg_e2_subj_id.cuda()
        neg_e2_subj_w = neg_e2_subj_w.cuda()
        neg_e2_verb_id = neg_e2_verb_id.cuda()
        neg_e2_verb_w = neg_e2_verb_w.cuda()
        neg_e2_obj_id = neg_e2_obj_id.cuda()
        neg_e2_obj_w = neg_e2_obj_w.cuda()

    pos_e1_emb = model(pos_e1_subj_id, pos_e1_subj_w, pos_e1_verb_id, pos_e1_verb_w, pos_e1_obj_id, pos_e1_obj_w)
    pos_e2_emb = model(pos_e2_subj_id, pos_e2_subj_w, pos_e2_verb_id, pos_e2_verb_w, pos_e2_obj_id, pos_e2_obj_w)
    neg_e1_emb = model(neg_e1_subj_id, neg_e1_subj_w, neg_e1_verb_id, neg_e1_verb_w, neg_e1_obj_id, neg_e1_obj_w)
    neg_e2_emb = model(neg_e2_subj_id, neg_e2_subj_w, neg_e2_verb_id, neg_e2_verb_w, neg_e2_obj_id, neg_e2_obj_w)

    cosine_similarity = nn.CosineSimilarity(dim=1)
    pos_sim = cosine_similarity(pos_e1_emb, pos_e2_emb)
    neg_sim = cosine_similarity(neg_e1_emb, neg_e2_emb)
    num_correct = (pos_sim > neg_sim).sum().item()
    accuracy = num_correct / len(dataset)

    if option.output_file.strip() != '':
        output_file = open(option.output_file, 'w')
        for i, j, k in zip(pos_sim, neg_sim, (pos_sim > neg_sim)):
            output_file.write(' '.join([str(i.item()), str(j.item()), str(k.item())]) + '\n')
        output_file.close()
        logging.info('Output saved to ' + option.output_file)

    logging.info('Num correct: ' + str(num_correct))
    logging.info('Num total: ' + str(len(dataset)))
    logging.info('Accuracy: ' + str(accuracy))
