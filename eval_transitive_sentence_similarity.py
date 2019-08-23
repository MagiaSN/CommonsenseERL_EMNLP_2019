import torch.nn as nn
import torch.utils.data
import scipy.stats
import sys
import logging
import argparse
from model import LowRankNeuralTensorNetwork, NeuralTensorNetwork, RoleFactoredTensorModel, PredicateTensorModel, Averaging, NN, EMC
from dataset import TransitiveSentenceSimilarityDataset, TransitiveSentenceSimilarityDataset_collate_fn
from event_tensors.glove_utils import Glove


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--vocab_size', type=int, default=400000)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--emb_file', type=str, default='data/glove.6B.100d.ext.txt')
    parser.add_argument('--dataset_file', type=str, default='data/transitive.txt')
    parser.add_argument('--model_file', type=str, default='model/nyt/ntn/NeuralTensorNetwork_2007.pt')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--model', type=str, default='NTN')
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--em_r', type=int, default=10)
    option = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    glove = Glove(option.emb_file)
    logging.info('Embeddings loaded')

    dataset = TransitiveSentenceSimilarityDataset(option.dataset_file, glove)
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

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=TransitiveSentenceSimilarityDataset_collate_fn, shuffle=False, batch_size=len(dataset))
    batch = next(iter(data_loader))
    e1_subj_id, e1_subj_w, e1_verb_id, e1_verb_w, e1_obj_id, e1_obj_w, \
    e2_subj_id, e2_subj_w, e2_verb_id, e2_verb_w, e2_obj_id, e2_obj_w, \
    gold = batch

    if option.use_gpu:
        e1_subj_id = e1_subj_id.cuda()
        e1_subj_w = e1_subj_w.cuda()
        e1_verb_id = e1_verb_id.cuda()
        e1_verb_w = e1_verb_w.cuda()
        e1_obj_id = e1_obj_id.cuda()
        e1_obj_w = e1_obj_w.cuda()
        e2_subj_id = e2_subj_id.cuda()
        e2_subj_w = e2_subj_w.cuda()
        e2_verb_id = e2_verb_id.cuda()
        e2_verb_w = e2_verb_w.cuda()
        e2_obj_id = e2_obj_id.cuda()
        e2_obj_w = e2_obj_w.cuda()

    e1_emb = model(e1_subj_id, e1_subj_w, e1_verb_id, e1_verb_w, e1_obj_id, e1_obj_w)
    e2_emb = model(e2_subj_id, e2_subj_w, e2_verb_id, e2_verb_w, e2_obj_id, e2_obj_w)
    cosine_similarity = nn.CosineSimilarity(dim=1)
    pred = cosine_similarity(e1_emb, e2_emb)

    if option.use_gpu:
        pred = pred.cpu()
    pred = pred.detach().numpy()
    gold = gold.numpy()
    spearman_correlation, spearman_p = scipy.stats.spearmanr(pred, gold)

    if option.output_file.strip() != '':
        output_file = open(option.output_file, 'w')
        for score in pred:
            output_file.write(str(score) + '\n')
        output_file.close()
        logging.info('Output saved to ' + option.output_file)

    logging.info('Spearman correlation: ' + str(spearman_correlation))
