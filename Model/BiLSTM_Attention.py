import torch
from torch import nn, optim
import torch.nn.functional as F
import random
import gensim.models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"

SEED = 1222

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sequence_to_tripeptide(file, label):
    peptide = []
    for line in open(file):
        line = line.strip()
        if line.startswith(">"):
            continue
        else:
            peptides = []
            for i in range(len(line) - 2):
                peptides.append(line[i:i + 3])

            peptide.append((label, peptides))

    return peptide


word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("seven_n3_word_noAll_sample_workers1.vector")
vector = torch.tensor(word2vec_model.vectors, dtype=torch.float)
word2vec_list = word2vec_model.index_to_key


def sequence_to_index(sequences, word_index):

    index_set = []
    for word in sequences:
        index_set.append(word_index.index(word))

    return torch.tensor(index_set, dtype=torch.long)


class LSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layer, bidirectional, dropout):
        super(LSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=bidirectional, dropout=dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layer, bidirectional=bidirectional, dropout=dropout)

        # decoder层
        if bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, output_dim)
        else:
            self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, output_dim)

        # dropout层
        self.dropout = nn.Dropout(dropout)

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat((final_state[-1, :, :], final_state[-2, :, :]), 1)
        merged_state = merged_state.unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, inputs):
        inputs = inputs.view(len(inputs), -1)
        embedded = self.dropout(self.embedding(inputs))
        output, (hidden, cell) = self.lstm(embedded)
        attn_output = self.dropout(self.attention(output, hidden))
        attn_output = self.decoder1(attn_output.squeeze(0))

        return self.decoder2(attn_output)


VOCAB_SIZE = len(word2vec_list)
OUTPUT_DIM = 2
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
DROPOUT = 0.5
NUM_LAYERS = 2
BIDIRECTIONAL = True

model = LSTM_Attention(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, BIDIRECTIONAL, DROPOUT).cuda()
model.embedding.weight.data.copy_(vector)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000005)
criterion = criterion.to(device)


def binary_accuracy(predictions, tag):

    rounded_predictions = torch.round(torch.sigmoid(predictions))
    correct = (rounded_predictions == tag).float()
    print("correct.sum():", correct.sum(), "correct_number:", len(correct))
    acc = correct.sum()/len(correct)
    return acc


def train(model, data, loss_function, optimizer):

    epoch_loss = 0
    epoch_auc = 0
    model.train()

    for label, sequence in data:
        optimizer.zero_grad()

        sequence_index = sequence_to_index(sequence, word2vec_list).cuda()
        target = torch.tensor([label], dtype=torch.long).cuda()
        label_scores = model(sequence_index)
        label_scores = label_scores.view(-1, len(label_scores))
        predict_index = torch.max(label_scores, 1)[1]
        auc = torch.as_tensor(predict_index == target, dtype=torch.float)
        loss = loss_function(label_scores, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_auc += auc.item()

    return epoch_auc / len(data), epoch_loss / len(data)


def evaluate(model, data, loss_function):

    average_loss = 0
    epoch_auc = 0
    model.eval()

    with torch.no_grad():
        for label, sequence in data:
            model.zero_grad()

            sequence_index = sequence_to_index(sequence, word2vec_list).cuda()
            target = torch.tensor([label], dtype=torch.long).cuda()
            label_scores = model(sequence_index)
            label_scores = label_scores.view(-1, len(label_scores))
            predict_idx = torch.max(label_scores, 1)[1]
            auc = torch.as_tensor(predict_idx == target, dtype=torch.float)
            loss = loss_function(label_scores, target)

            average_loss += loss.item()
            epoch_auc += auc.item()

    return epoch_auc / len(data), average_loss / len(data)


train_positive_file = "positive_seven_train_sample.txt"
train_negative_file = "negative_sevenN3_train_sample.txt"
test_positive_file = 'positive_seven_test_sample.txt'
test_negative_file = 'negative_sevenN3_test_sample.txt'

train_data = sequence_to_tripeptide(train_positive_file, 1) + sequence_to_tripeptide(train_negative_file, 0)
test_data = sequence_to_tripeptide(test_positive_file, 1) + sequence_to_tripeptide(test_negative_file, 0)
random.shuffle(test_data)
random.shuffle(train_data)

best_acc = 0

for epoch in range(1000):

    # if epoch == 0:
    #     model.load_state_dict(torch.load('LSTM-model-.pt'))
    train_auc, train_loss = train(model, train_data, criterion, optimizer)
    print("epoch:", epoch, "train_auc:", train_auc, "train_loss:", train_loss)
    evaluate_auc, evaluate_loss = evaluate(model, test_data, criterion)
    print("epoch:", epoch, "evaluate_auc:", evaluate_auc, "evaluate_loss:", evaluate_loss)

    if evaluate_auc > best_acc:
        torch.save(model.state_dict(), 'BiLSTM_Attention-model-seven_n3_word_noAll_sample_workers1.pt')
        best_acc = evaluate_auc
        print('best_acc:', best_acc)
