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


def sequence_to_index(sequences, word_index):
    index_set = []
    for word in sequences:
        index_set.append(word_index.index(word))

    return torch.tensor(index_set, dtype=torch.long)


word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("seven_n3_word_noAll_sample_workers1.vector")
vector = torch.tensor(word2vec_model.vectors, dtype=torch.float)
word2vec_list = word2vec_model.index_to_key


class CNN_BiLSTM(nn.Module):

    def __init__(self, vocab_dim, embedding_dim, n_filters, filter_sizes, hidden_dim, output_dim, num_layers,
                 bidirectional, dropout):
        super(CNN_BiLSTM, self).__init__()
        # self.args = args
        # self.hidden_dim = args.lstm_hidden_dim
        # self.num_layers = args.lstm_num_layers
        # V = args.embed_num
        # D = args.embed_dim
        # C = args.class_num
        # self.C = C
        # Ci = 1
        # Co = args.kernel_num
        # Ks = args.kernel_sizes
        self.embed = nn.Embedding(vocab_dim, embedding_dim)

        # CNN
        self.convs1 = [
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(K, embedding_dim), padding=(K // 2, 0),
                      stride=1) for K in filter_sizes]
        # for cnn cuda
        if torch.cuda.is_available():
            for conv in self.convs1:
                conv = conv.cuda()

        # BiLSTM
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout,
                              bidirectional=bidirectional)

        # linear
        L = len(filter_sizes) * n_filters + hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, output_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat((final_state[-1, :, :], final_state[-2, :, :]), 1)
        merged_state = merged_state.unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        # print(weights.shape)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, inputs):
        inputs = inputs.view(len(inputs), -1)
        embed = self.dropout(self.embed(inputs))

        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        # print("cnn_x:", cnn_x.shape)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        # print("cnn_x:",  len(cnn_x))
        cnn_x = [torch.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        # print("cnn_x:", len(cnn_x))
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)
        # print("cnn_x:", cnn_x.shape)

        # BiLSTM
        bilstm_x = embed.view(len(inputs), embed.size(1), -1)
        bilstm_out, (hidden, cell) = self.bilstm(bilstm_x)
        bilstm_out = self.attention(bilstm_out, hidden)
        # bilstm_out = self.dropout(bilstm_x)
        # print("bilstm_out:", bilstm_out.shape)
        # print("hidden:", hidden.shape)
        # bilstm_out = torch.transpose(bilstm_out, 0, 1)
        # bilstm_out = torch.transpose(bilstm_out, 1, 2)
        # bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        # bilstm_out = torch.tanh(bilstm_out)
        # print("bilstm_out:", bilstm_out.shape)

        # CNN and BiLSTM CAT
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = self.dropout(torch.cat((cnn_x, bilstm_out), 0))
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)
        # print("cnn_bilstm_out:", cnn_bilstm_out.shape)
        # print("------------------------------")

        # linear
        cnn_bilstm_out = self.hidden2label1(torch.tanh(cnn_bilstm_out))
        cnn_bilstm_out = self.hidden2label2(torch.tanh(cnn_bilstm_out))

        # output
        logit = cnn_bilstm_out
        return logit


# 定义超参数
# embed
VOCAB_SIZE = len(word2vec_list)
EMBEDDING_DIM = 100
# CNN
n_filters = 100
filter_sizes = [2, 3, 4]
# RNN
OUTPUT_DIM = 2
HIDDEN_DIM = 576
DROPOUT = 0.5
NUM_LAYERS = 2
BIDIRECTIONAL = True

model = CNN_BiLSTM(VOCAB_SIZE, EMBEDDING_DIM, n_filters, filter_sizes, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, BIDIRECTIONAL, DROPOUT).cuda()  # 需要修改
model.embed.weight.data.copy_(vector)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000008)
criterion = criterion.to(device)


def binary_accuracy(predictions, tag):
    rounded_predictions = torch.round(torch.sigmoid(predictions))
    correct = (rounded_predictions == tag).float()
    print("correct.sum():", correct.sum(), "correct_number:", len(correct))
    acc = correct.sum() / len(correct)
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
        # print(label_scores)
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
        torch.save(model.state_dict(), 'CNN-BiLSTM-Attention-model-seven_n3_word_noAll_sample_workers1_hidden_dim576_lr000008.pt')
        best_acc = evaluate_auc
        print('best_acc:', best_acc)
