import torch
import torch.nn as nn
import torch.utils.data
import logging
import os
import sys
import random
import argparse
from model import NeuralTensorNetwork, RoleFactoredTensorModel
from dataset import EmbeddingWithBias, WordPredictionDataset, WordPredictionDataset_collate_fn
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
    parser.add_argument('--dataset_file', type=str, default='data/word_prediction_small.txt')
    parser.add_argument('--model', type=str, default='RoleFactor')
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--em_r', type=int, default=10)
    parser.add_argument('--neg_samples', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--initial_accumulator_value', type=float, default=0.1)
    parser.add_argument('--report_every', type=int, default=200)
    parser.add_argument('--save_checkpoint', type=str, default='')
    parser.add_argument('--load_checkpoint', type=str, default='')
    option = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    torch.manual_seed(option.random_seed)
    random.seed(option.random_seed)

    glove = Glove(option.emb_file)
    logging.info('Embeddings loaded')

    embeddings = nn.Embedding(option.vocab_size, option.emb_dim, padding_idx=1)
    neg_embeddings = EmbeddingWithBias(option.vocab_size, option.emb_dim)
    if option.model == 'NTN':
        event_model = NeuralTensorNetwork(embeddings, option.em_k)
    elif option.model == 'RoleFactor':
        event_model = RoleFactoredTensorModel(embeddings, option.em_k)
    else:
        logging.info('Unknwon model: ' + option.model)
        exit(1)
    criterion = nn.CrossEntropyLoss()

    # load pretrained embeddings
    embeddings.weight.data.copy_(torch.from_numpy(glove.embd).float())

    if not option.update_embeddings:
        event_model.embeddings.weight.requires_grad = False

    if option.use_gpu:
        event_model.cuda()
        neg_embeddings.cuda()
        criterion.cuda()

    params = [
        { 'params': event_model.embeddings.parameters() },
        { 'params': neg_embeddings.parameters() }
    ]
    if option.model == 'NTN':
        params += [
            { 'params': event_model.subj_verb_comp.parameters(), 'weight_decay': option.weight_decay },
            { 'params': event_model.verb_obj_comp.parameters(), 'weight_decay': option.weight_decay },
            { 'params': event_model.final_comp.parameters(), 'weight_decay': option.weight_decay },
            { 'params': event_model.linear1.parameters(), 'weight_decay': option.weight_decay },
            { 'params': event_model.linear2.parameters(), 'weight_decay': option.weight_decay },
            { 'params': event_model.linear3.parameters(), 'weight_decay': option.weight_decay }
        ]
    elif option.model == 'RoleFactor':
        params += [
            { 'params': event_model.tensor_comp.parameters(), 'weight_decay': option.weight_decay },
            { 'params': event_model.w.parameters(), 'weight_decay': option.weight_decay }
        ]
    else:
        params = None

    optimizer = torch.optim.Adagrad(params, lr=option.lr, initial_accumulator_value=option.initial_accumulator_value)
    # load checkpoint if provided
    if option.load_checkpoint != '':
        checkpoint = torch.load(option.load_checkpoint)
        event_model.load_state_dict(checkpoint['model_state_dict'])
        neg_embeddings.load_state_dict(checkpoint['neg_embeddings_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info('Loaded checkpoint: ' + option.load_checkpoint)

    dataset = WordPredictionDataset()
    logging.info('Loading dataset: ' + option.dataset_file)
    dataset.load(option.dataset_file, glove)
    logging.info('Loaded dataset: ' + option.dataset_file)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=WordPredictionDataset_collate_fn, batch_size=option.batch_size, shuffle=False)

    for epoch in range(option.epochs):
        epoch += 1
        logging.info('Epoch ' + str(epoch))

        avg_loss = 0
        for i, batch in enumerate(data_loader):
            i += 1
            optimizer.zero_grad()

            subj_id, subj_w, verb_id, verb_w, obj_id, obj_w, word_id = batch
            batch_size = word_id.size(0)
            neg_samples = torch.LongTensor(random.sample(range(option.vocab_size), batch_size * option.neg_samples)).view(batch_size, -1)
            word_id = torch.cat([word_id.unsqueeze(1), neg_samples], dim=1)
            labels = torch.zeros(batch_size).long()

            if option.use_gpu:
                subj_id = subj_id.cuda()
                subj_w = subj_w.cuda()
                verb_id = verb_id.cuda()
                verb_w = verb_w.cuda()
                obj_id = obj_id.cuda()
                obj_w = obj_w.cuda()
                word_id = word_id.cuda()
                labels = labels.cuda()

            event_emb = event_model(subj_id, subj_w, verb_id, verb_w, obj_id, obj_w)    # (batch, emb_dim)
            nce_weights, nce_biases = neg_embeddings(word_id)   # (batch, 1+neg, emb_dim), (batch, 1+neg)
            scores = torch.bmm(
                event_emb.unsqueeze(1),         # (batch, 1, emb_dim)
                nce_weights.transpose(1, 2)     # (batch, emb_dim, 1+neg)
            ).squeeze() + nce_biases            # (batch, 1+neg)
            loss = criterion(scores, labels)

            avg_loss += loss.item() / option.report_every
            if i % option.report_every == 0:
                logging.info('Batch %d, loss=%.4f' % (i, avg_loss))
                avg_loss = 0

    if option.save_checkpoint != '':
        checkpoint = {
            'model_state_dict': event_model.state_dict(),
            'neg_embeddings_state_dict': neg_embeddings.state_dict(),
            'optimizer_staet_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, option.save_checkpoint)
        logging.info('Saved checkpoint: ' + option.save_checkpoint)
