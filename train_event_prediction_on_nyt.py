import torch
import torch.nn as nn
import torch.utils.data
import logging
import os
import sys
import argparse
from model import NeuralTensorNetwork, LowRankNeuralTensorNetwork, RoleFactoredTensorModel, PredicateTensorModel, MarginLoss, NN, EMC
from dataset import EventPredictionDataset, EventPredictionDataset_collate_fn
from event_tensors.glove_utils import Glove


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=19950125)
    parser.add_argument('--vocab_size', type=int, default=400000)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--update_embeddings', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--emb_file', type=str, default='data/glove.6B.100d.ext.txt')
    parser.add_argument('--dataset_file', type=str, default='data/event_prediction/1987.txt')
    parser.add_argument('--model', type=str, default='NTN')
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--em_r', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--initial_accumulator_value', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--report_every', type=int, default=1000)
    parser.add_argument('--save_checkpoint', type=str, default='')
    parser.add_argument('--load_checkpoint', type=str, default='')
    option = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    torch.manual_seed(option.random_seed)

    glove = Glove(option.emb_file)
    logging.info('Embeddings loaded')

    embeddings = nn.Embedding(option.vocab_size, option.emb_dim, padding_idx=1)
    if option.model == 'NTN':
        event_model = NeuralTensorNetwork(embeddings, option.em_k)
    elif option.model == 'LowRankNTN':
        event_model = LowRankNeuralTensorNetwork(embeddings, option.em_k, option.em_r)
    elif option.model == 'RoleFactor':
        event_model = RoleFactoredTensorModel(embeddings, option.em_k)
    elif option.model == 'Predicate':
        event_model = PredicateTensorModel(embeddings)
    elif option.model == 'NN':
        event_model = NN(embeddings, 2 * option.em_k, option.em_k)
    elif option.model == 'EMC':
        event_model = EMC(embeddings, 2 * option.em_k, option.em_k)
    else:
        logging.info('Unknwon model: ' + option.model)
        exit(1)
    cosine_similarity = nn.CosineSimilarity(dim=1)
    criterion = MarginLoss(option.margin)

    # load pretrained embeddings
    embeddings.weight.data.copy_(torch.from_numpy(glove.embd).float())

    if not option.update_embeddings:
        event_model.embeddings.weight.requires_grad = False

    if option.use_gpu:
        event_model.cuda()
        cosine_similarity.cuda()
        criterion.cuda()

    embedding_param_id = [id(param) for param in embeddings.parameters()]
    params = [
        { 'params': embeddings.parameters() },
        { 'params': [param for param in event_model.parameters() if id(param) not in embedding_param_id], 'weight_decay': option.weight_decay }
    ]
    optimizer = torch.optim.Adagrad(params, lr=option.lr, initial_accumulator_value=option.initial_accumulator_value)

    # load checkpoint if provided
    if option.load_checkpoint != '':
        checkpoint = torch.load(option.load_checkpoint)
        event_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info('Loaded checkpoint: ' + option.load_checkpoint)

    dataset = EventPredictionDataset()
    logging.info('Loading dataset: ' + option.dataset_file)
    dataset.load(option.dataset_file, glove)
    # dataset.data = dataset.data[:786432]
    logging.info('Loaded dataset: ' + option.dataset_file)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=EventPredictionDataset_collate_fn, batch_size=option.batch_size, shuffle=False)

    for epoch in range(option.epochs):
        epoch += 1
        logging.info('Epoch ' + str(epoch))

        avg_loss = 0
        for i, batch in enumerate(data_loader):
            i += 1
            optimizer.zero_grad()

            ei_subj_id, ei_subj_w, ei_verb_id, ei_verb_w, ei_obj_id, ei_obj_w, \
            et_subj_id, et_subj_w, et_verb_id, et_verb_w, et_obj_id, et_obj_w, \
            en_subj_id, en_subj_w, en_verb_id, en_verb_w, en_obj_id, en_obj_w = batch

            if option.use_gpu:
                ei_subj_id = ei_subj_id.cuda()
                ei_subj_w = ei_subj_w.cuda()
                ei_verb_id = ei_verb_id.cuda()
                ei_verb_w = ei_verb_w.cuda()
                ei_obj_id = ei_obj_id.cuda()
                ei_obj_w = ei_obj_w.cuda()
                et_subj_id = et_subj_id.cuda()
                et_subj_w = et_subj_w.cuda()
                et_verb_id = et_verb_id.cuda()
                et_verb_w = et_verb_w.cuda()
                et_obj_id = et_obj_id.cuda()
                et_obj_w = et_obj_w.cuda()
                en_subj_id = en_subj_id.cuda()
                en_subj_w = en_subj_w.cuda()
                en_verb_id = en_verb_id.cuda()
                en_verb_w = en_verb_w.cuda()
                en_obj_id = en_obj_id.cuda()
                en_obj_w = en_obj_w.cuda()

            ei_emb = event_model(ei_subj_id, ei_subj_w, ei_verb_id, ei_verb_w, ei_obj_id, ei_obj_w)
            et_emb = event_model(et_subj_id, et_subj_w, et_verb_id, et_verb_w, et_obj_id, et_obj_w)
            en_emb = event_model(en_subj_id, en_subj_w, en_verb_id, en_verb_w, en_obj_id, en_obj_w)
            pos_score = cosine_similarity(ei_emb, et_emb)
            neg_score = cosine_similarity(ei_emb, en_emb)
            loss = criterion(pos_score, neg_score)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / option.report_every
            if i % option.report_every == 0:
                logging.info('Batch %d, loss=%.4f' % (i, avg_loss))
                avg_loss = 0

        if option.save_checkpoint != '':
            checkpoint = {
                'model_state_dict': event_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, option.save_checkpoint + '_' + str(epoch))
            logging.info('Saved checkpoint: ' + option.save_checkpoint + '_' + str(epoch))
