import torch
import torch.nn as nn
import torch.utils.data
import sys
import logging
import argparse
from model import NeuralTensorNetwork, LowRankNeuralTensorNetwork, RoleFactoredTensorModel, BiLSTMEncoder, MarginLoss
from dataset import EventIntentSentimentDataset, EventIntentSentimentDataset_collate_fn
from event_tensors.glove_utils import Glove


def run_batch(option, batch, event_model, intent_model, event_scorer, intent_scorer, sentiment_classifier, criterion, sentiment_criterion):
    subj_id, subj_w, verb_id, verb_w, obj_id, obj_w, neg_obj_id, neg_obj_w, \
    intent, neg_intent, sentiment, neg_sentiment = batch
    if option.use_gpu:
        subj_id = subj_id.cuda()
        subj_w = subj_w.cuda()
        verb_id = verb_id.cuda()
        verb_w = verb_w.cuda()
        obj_id = obj_id.cuda()
        obj_w = obj_w.cuda()
        neg_obj_id = neg_obj_id.cuda()
        neg_obj_w = neg_obj_w.cuda()
        intent = intent.cuda()
        neg_intent = neg_intent.cuda()
        sentiment = sentiment.cuda()
        neg_sentiment = neg_sentiment.cuda()
    # event loss
    pos_event_emb = event_model(subj_id, subj_w, verb_id, verb_w, obj_id, obj_w)
    neg_event_emb = event_model(subj_id, subj_w, verb_id, verb_w, neg_obj_id, neg_obj_w)
    pos_event_score = event_scorer(pos_event_emb).squeeze()
    neg_event_score = event_scorer(neg_event_emb).squeeze()
    loss_e = criterion(pos_event_score, neg_event_score)
    # intent loss
    pos_intent_emb = intent_model(intent)
    neg_intent_emb = intent_model(neg_intent)
    pos_intent_score = intent_scorer(pos_event_emb, pos_intent_emb)
    neg_intent_score = intent_scorer(pos_event_emb, neg_intent_emb)
    loss_i = criterion(pos_intent_score, neg_intent_score)
    # sentiment loss
    sentiment_pred = sentiment_classifier(pos_event_emb).squeeze()
    loss_s = sentiment_criterion(sentiment_pred, sentiment.float())
    loss = option.alpha1 * loss_e + option.alpha2 * loss_i + option.alpha3 * loss_s
    return loss, loss_e, loss_i, loss_s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=19950125)
    parser.add_argument('--vocab_size', type=int, default=400000)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--update_embeddings', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--emb_file', type=str, default='data/glove.6B.100d.ext.txt')
    parser.add_argument('--train_dataset', type=str, default='data/atomic/all2.txt')
    parser.add_argument('--dev_dataset', type=str, default='data/atomic/dev2.txt')
    parser.add_argument('--model', type=str, default='NTN')
    parser.add_argument('--pretrained_event_model', type=str, default='model/nyt/ntn/NeuralTensorNetwork_2007.pt')
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--em_r', type=int, default=10)
    parser.add_argument('--em_actv_func', type=str, default='sigmoid')
    parser.add_argument('--im_hidden_size', type=int, default=100)
    parser.add_argument('--im_num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--alpha1', type=float, default=0.33)
    parser.add_argument('--alpha2', type=float, default=0.33)
    parser.add_argument('--alpha3', type=float, default=0.33)
    parser.add_argument('--report_every', type=int, default=50)
    parser.add_argument('--load_checkpoint', type=str, default='')
    parser.add_argument('--save_checkpoint', type=str, default='')
    option = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    torch.manual_seed(option.random_seed)

    glove = Glove(option.emb_file)
    logging.info('Embeddings loaded')

    embeddings = nn.Embedding(option.vocab_size, option.emb_dim, padding_idx=1)
    if option.model == 'NTN':
        event_model = NeuralTensorNetwork(embeddings, option.em_k)
    elif option.model == 'RoleFactor':
        event_model = RoleFactoredTensorModel(embeddings, option.em_k)
    elif option.model == 'LowRankNTN':
        event_model = LowRankNeuralTensorNetwork(embeddings, option.em_k, option.em_r)
    else:
        logging.info('Unknown model type: ' + option.model)
        exit(1)
    intent_model = BiLSTMEncoder(embeddings, option.im_hidden_size, option.im_num_layers)

    if option.em_actv_func == 'sigmoid':
        em_actv_func = nn.Sigmoid()
    elif option.em_actv_func == 'relu':
        em_actv_func = nn.ReLU()
    elif option.em_actv_func == 'tanh':
        em_actv_func = nn.Tanh()
    else:
        logging.info('Unknown event activation func: ' + option.em_actv_func)
        exit(1)
    event_scorer = nn.Sequential(nn.Linear(option.em_k, 1), em_actv_func)

    intent_scorer = nn.CosineSimilarity(dim=1)
    sentiment_classifier = nn.Linear(option.em_k, 1)
    criterion = MarginLoss(option.margin)
    sentiment_criterion = nn.BCEWithLogitsLoss()

    # load pretrained embeddings
    embeddings.weight.data.copy_(torch.from_numpy(glove.embd).float())

    if not option.update_embeddings:
        event_model.embeddings.weight.requires_grad = False

    if option.use_gpu:
        event_model.cuda()
        intent_model.cuda()
        sentiment_classifier.cuda()
        event_scorer.cuda()

    embeddings_param_id = [id(param) for param in embeddings.parameters()]
    params = [
        { 'params': embeddings.parameters() },
        { 'params': [param for param in event_model.parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay },
        { 'params': [param for param in event_scorer.parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay },
        { 'params': [param for param in intent_model.parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay },
        { 'params': [param for param in sentiment_classifier.parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay }
    ]
    optimizer = torch.optim.Adagrad(params, lr=option.lr)

    # load checkpoint if provided:
    if option.load_checkpoint != '':
        checkpoint = torch.load(option.load_checkpoint)
        event_model.load_state_dict(checkpoint['event_model_state_dict'])
        intent_model.load_state_dict(checkpoint['intent_model_state_dict'])
        event_scorer.load_state_dict(checkpoint['event_scorer_state_dict'])
        sentiment_classifier.load_state_dict(checkpoint['sentiment_classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info('Loaded checkpoint: ' + option.load_checkpoint)
    # load pretrained event model instead:
    elif option.pretrained_event_model != '':
        checkpoint = torch.load(option.pretrained_event_model)
        event_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('Loaded pretrained event model: ' + option.pretrained_event_model)

    train_dataset = EventIntentSentimentDataset()
    logging.info('Loading train dataset: ' + option.train_dataset)
    train_dataset.load(option.train_dataset, glove)
    logging.info('Loaded train dataset: ' + option.train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=EventIntentSentimentDataset_collate_fn, batch_size=option.batch_size, shuffle=True)

    dev_dataset = EventIntentSentimentDataset()
    logging.info('Loading dev dataset: ' + option.dev_dataset)
    dev_dataset.load(option.dev_dataset, glove)
    logging.info('Loaded dev dataset: ' + option.dev_dataset)
    dev_data_loader = torch.utils.data.DataLoader(dev_dataset, collate_fn=EventIntentSentimentDataset_collate_fn, batch_size=len(dev_dataset), shuffle=False)

    for epoch in range(option.epochs):
        epoch += 1
        logging.info('Epoch ' + str(epoch))

        # train set
        avg_loss_e = 0
        avg_loss_i = 0
        avg_loss_s = 0
        avg_loss = 0
        for i, batch in enumerate(train_data_loader):
            i += 1
            optimizer.zero_grad()
            loss, loss_e, loss_i, loss_s = run_batch(option, batch, event_model, intent_model, event_scorer, intent_scorer, sentiment_classifier, criterion, sentiment_criterion)
            loss.backward()
            optimizer.step()

            avg_loss_e += loss_e.item() / option.report_every
            avg_loss_i += loss_i.item() / option.report_every
            avg_loss_s += loss_s.item() / option.report_every
            avg_loss += loss.item() / option.report_every
            if i % option.report_every == 0:
                logging.info('Batch %d, loss_e=%.4f, loss_i=%.4f, loss_s=%.4f, loss=%.4f' % (i, avg_loss_e, avg_loss_i, avg_loss_s, avg_loss))
                avg_loss_e = 0
                avg_loss_i = 0
                avg_loss_s = 0
                avg_loss = 0

        # dev set
        event_model.eval()
        intent_model.eval()
        event_scorer.eval()
        sentiment_classifier.eval()
        batch = next(iter(dev_data_loader))
        loss, loss_e, loss_i, loss_s = run_batch(option, batch, event_model, intent_model, event_scorer, intent_scorer, sentiment_classifier, criterion, sentiment_criterion)
        logging.info('Eval on dev set, loss_e=%.4f, loss_i=%.4f, loss_s=%.4f, loss=%.4f' % (loss_e.item(), loss_i.item(), loss_s.item(), loss.item()))
        event_model.train()
        intent_model.train()
        event_scorer.train()
        sentiment_classifier.train()

        if option.save_checkpoint != '':
            checkpoint = {
                'event_model_state_dict': event_model.state_dict(),
                'intent_model_state_dict': intent_model.state_dict(),
                'event_scorer_state_dict': event_scorer.state_dict(),
                'sentiment_classifier_state_dict': sentiment_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, option.save_checkpoint + '_' + str(epoch))
            logging.info('Saved checkpoint: ' + option.save_checkpoint + '_' + str(epoch))
