#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import pandas as pd
import json
import math
import torch
import string
from random import randint
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

import random
import tqdm
import sys

# In[2]:

DATA_DIR = sys.argv[1]
output_file = sys.argv[2]


# In[3]:


train_label_file=f"{DATA_DIR}/training_label.json"
test_label_file=f"{DATA_DIR}/testing_label.json"
test_numpy_feat=f"{DATA_DIR}/testing_data/feat/"
train_numpy_feat=f"{DATA_DIR}/training_data/feat/"


# In[4]:


train_size=1400


# In[5]:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[6]:


class VocabBuilder:
    def __init__(self,labels,MIN_FREQ=3):
        self.word2index = {"<BOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
        self.index2word = {0: "<BOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>"}
        self.vocabulary = Counter()
        self.vocab_size = 0
        self.min_count = MIN_FREQ
        # construct vocabularly, then construct word2id/id2word
        self.construct_vocab(labels)
        self.construct_word2id()
    
    def construct_vocab(self, labels):
        for index,rows in labels.iterrows():
            for caption in np.unique(np.array(rows['caption'])):
                caption_word_sent= ' '.join(word.strip(string.punctuation).lower() for word in caption.split())

                words_list=caption_word_sent.split(" ")

                self.vocabulary+=Counter(words_list)
        self.vocabulary = Counter({vocab:count for vocab, count in self.vocabulary.items() if count > self.min_count})
        self.vocab_size = len(self.vocabulary) + 4
        
    def construct_word2id(self):
        # word index start at 4
        word_index = 4
        # construct word2index, index2word
        for word in self.vocabulary:
            self.word2index[word] = word_index
            self.index2word[word_index] = word
            word_index += 1   


# In[7]:


MAXLENGTHOFSENTENCE=21


# In[8]:


class VidCaptionDataset(Dataset):
    def __init__ (self, train_label_file):
        self.video_labels = pd.read_json(train_label_file).set_index('id')
        self.captions_list = []
        self.vocab = VocabBuilder(self.video_labels)
        self.video_features_dict = {}
        # preprocess
        self.preprocess()
        
    def preprocess(self):
       
        counter = 0
        for index, row in self.video_labels[:train_size].iterrows():
            self.video_features_dict[index] = torch.FloatTensor(np.load(f"{train_numpy_feat}{index}.npy"))
#             print(self.video_features_dict[index].shape)
            # tokenize caption
            for caption in np.unique(np.array(row['caption'])):
                new_caption = []
                for word in caption.split():
                    word = word.strip(string.punctuation).lower()
                    if word in self.vocab.vocabulary:
                        new_caption.append(word)
                    else:
                        new_caption.append("<UNK>")
                if (len(new_caption) + 1) > MAXLENGTHOFSENTENCE:
                    continue
                # Number of padding to be appended
                cap_len = MAXLENGTHOFSENTENCE - (len(new_caption) + 1)
                new_caption += ["<EOS>"]
                new_caption += ["<PAD>"] * cap_len
                new_caption = ' '.join(new_caption)
                self.captions_list.append([new_caption, index])

    def __len__(self):
        return len(self.captions_list)
    
    def __getitem__(self, idx):
        caption, vid = self.captions_list[idx]
        feature = self.video_features_dict[vid]
        cap_seprate = caption.split(" ")
        new_caps = [self.vocab.word2index[word] for word in cap_seprate]
        new_caps = torch.LongTensor(new_caps).view(MAXLENGTHOFSENTENCE, 1)
        cap_onehot = torch.LongTensor(MAXLENGTHOFSENTENCE, self.vocab.vocab_size)
        cap_onehot.zero_()
        cap_onehot.scatter_(1, new_caps, 1)
        sample = {'frame': feature, 'onehot': cap_onehot, 'caption': caption}
        return sample


# In[9]:
video_caption_training_dataset=[]
def create_vocab():
    video_caption_training_dataset = VidCaptionDataset(train_label_file)
    data_size = len(video_caption_training_dataset)
    VOCAB_SIZE = video_caption_training_dataset.vocab.vocab_size
    print("data size: %d, vocab size: %d" % (data_size, VOCAB_SIZE))




# In[12]:


class Seq2SeqVideoModel(nn.Module):
    #Here the video_step, is the number of frame which we are reading, we have data for every 80th frame so using that here
    #output_step is the size of expected output, here this will be equal to the maximum sentence we allow it to be.
    def __init__(self, feature_size,vocab_size,hidden_size,video_step,output_step,batch_size,n_layers=1,dropout=0.3):
        super(Seq2SeqVideoModel, self).__init__()
        #80
        self.video_step = video_step
        #MaxLengthOfSentence
        self.output_step = output_step
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        #This is 4096, because that is the size of the features we get from the numpy datasets.

        self.feature_size = feature_size
        self.embedding_size = 512
     
        
        self.attention = Attention(batch_size, hidden_size)
        #The embedding layers feeds into the encoder
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(feature_size, 512)
        #Takes input from embedder
        self.encoder = nn.GRU(512, hidden_size, n_layers, dropout=dropout)
        #Because of attention
        self.decoder = nn.GRU(hidden_size*2+self.embedding_size, hidden_size, n_layers, dropout=dropout)
        #output provides distribution over all of the vocab
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, video_seq, cap_seq, teacher_forcing_ratio):
        loss = 0
        # pad MAXLEN, batch, 4096
        padding_encoder =  Variable(torch.zeros(self.output_step, self.batch_size, 512)).to(device);
        # pad 80, batch, 256
        #contains captions as input, the features are fed from the last encoder
        #Here we are also passing each caption generated through a word embeddeding so last dimension is self.hidden_size+self.embedding_size
        padding_decoder = Variable(torch.zeros(self.video_step, self.batch_size, self.hidden_size+self.embedding_size)).to(device);
        BOS = [0] * self.batch_size
        BOS =Variable(torch.LongTensor([BOS])).resize(batch_size, 1).to(device)
        BOS = self.embedding(BOS)
        
        video_seq = self.dropout(F.selu(self.fc1(video_seq)))
        #the encoder takes video_seq+decoder_input_len
        encoder_input = torch.cat((video_seq, padding_encoder), 0)
        # output1:  (seq_len, batch, hidden_size)
        #the output contains all the hidden layers(https://stackoverflow.com/a/48305882/6063389).
        first_encoder_output, first_encoder_hidden = self.encoder(encoder_input)
        
        # cap_seq: batch, MAXLEN => batch, MAXLEN, hidden_size
        cap_embedded = self.embedding(cap_seq)
        #Here the decoder GRU takes the Padded data(this is equal to the size of encoder)
        first_decoder_input = torch.cat((padding_decoder, first_encoder_output[:self.video_step,:,:]),2)
        #We feed the decoder with end to get hidden state 
        first_decoder_output, first_decoder_hidden = self.decoder(first_decoder_input)
        z = first_decoder_hidden
        #Till Here all the input video frames have been exhausted and encoder output is fed into decoder
        # decoder of input video frames starting
        for step in range(self.output_step):
            use_teacher_forcing = True if random.random() <= teacher_forcing_ratio else False
            if step == 0:
                #In the starting BOS is fed to start decoding
                decoder_input = BOS
            elif use_teacher_forcing:
                #actual truth of the last time stamp is used as a input
                decoder_input = cap_embedded[:,step-1,:].unsqueeze(1)
            else:
                decoder_input = decoder_output.max(1)[-1].resize(batch_size, 1)
                decoder_input = self.embedding(decoder_input)
            #sending all the hidden states at every time to attention for scoring and context vector s
            attention_weights = self.attention('dot', z, first_encoder_output[:self.video_step])
            #batch matrix multiplication
            context = torch.bmm(attention_weights.transpose(1,2),
                                 first_encoder_output[:self.video_step].transpose(0,1))
            # batch, 1, hidden_size*2 
            #attention layer processing
            second_decoder_input = torch.cat((decoder_input, first_encoder_output[self.video_step+step].unsqueeze(1), context),2).transpose(0,1)
            
            decoder_output, z = self.decoder(second_decoder_input, z)
            decoder_output = self.softmax(self.out(decoder_output[0]))
            
            loss += F.nll_loss(decoder_output, cap_seq[:,step])
        return loss
    
#     def attn_regularization(self, attention):
#         time_sum = attention.sum(2)
#         tao = time_sum.mean(1).resize(self.batch_size, 1).expand(self.batch_size, time_sum.size()[1])
#         reg = torch.pow((tao - time_sum), 2).sum()
#         return reg
    
    def testing(self, video_seq, index2word,beam_search, beam_size):
        pred = []
        padding_encoder= Variable(torch.zeros(self.output_step, 1, 512)).to(device);
        padding_decoder = Variable(torch.zeros(self.video_step, 1, self.hidden_size+self.embedding_size)).to(device);
        BOS = [0]
        BOS = Variable(torch.LongTensor([BOS])).resize(1, 1).cuda()
        BOS = self.embedding(BOS)
        
        
        video_seq = F.selu(self.fc1(video_seq))
        
        encoder_input = torch.cat((video_seq, padding_encoder), 0)
        first_encoder_output, first_encoder_hidden = self.encoder(encoder_input)
        
        decoder_input = torch.cat((padding_decoder, first_encoder_output[:self.video_step,:,:]),2)
        first_decoder_output, first_decoder_hidden = self.decoder(decoder_input)
        z = first_decoder_hidden
        
        if beam_search:
            for step in range(self.output_step):
                if step == 0:
                    attention_weights = self.attention('dot', z, first_decoder_output[:self.video_step])
                    context = torch.bmm(attention_weights.transpose(1,2),
                                         first_decoder_output[:self.video_step].transpose(0,1))
                    # batch, 1, hidden_size*2
                    second_input_decoder = torch.cat((BOS, first_encoder_output[self.video_step+step].unsqueeze(1), context),2).transpose(0,1)

                    decoder_output, z = self.decoder(second_input_decoder, z)
                    decoder_output = self.softmax(self.out(decoder_output[0]))

                    softmax_probability = math.e ** decoder_output

                    top_candidate, indices_top = softmax_probability.topk(beam_size)
                    cur_scores = top_candidate.data[0].cpu().numpy().tolist()
                    candidates = indices_top.data[0].cpu().numpy().reshape(beam_size, 1).tolist()
                    zs = [z] * beam_size
                else:
                    new_candidates = []
                    for j, candidate in enumerate(candidates):
                        decoder_input = Variable(torch.LongTensor([candidate[-1]])).to(device).resize(1,1)
                        decoder_input = self.embedding(decoder_input)

                        attention_weights = self.attention('dot', z, first_encoder_output[:self.video_step])
                        context = torch.bmm(attention_weights.transpose(1,2),
                                             first_encoder_output[:self.video_step].transpose(0,1))
                        # batch, 1, hidden_size*2
                        second_input_decoder = torch.cat((decoder_input, first_encoder_output[self.video_step+step].unsqueeze(1), context),2).transpose(0,1)
                        decoder_output, zs[j] = self.decoder(second_input_decoder, zs[j])
                        decoder_output = self.softmax(self.out(decoder_output[0]))

                        softmax_prob = math.e ** decoder_output
                        top_candidate, indices_top = softmax_prob.topk(beam_size)
                        for k in range(beam_size):
                            score = cur_scores[j] * top_candidate.data[0, k]
                            new_candidate = candidates[j] + [indices_top.data[0, k]]
                            new_candidates.append([score, new_candidate, zs[j]])
                    # get top-k candidates and drop others
                    new_candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
                    cur_scores = [candi[0] for candi in new_candidates]
                    candidates = [candi[1] for candi in new_candidates]
                    zs = [cand[2] for cand in new_candidates]
            pred = [index2word[int(word_index)] for word_index in candidates[0] if int(word_index) >= 3]

        else:
            for step in range(self.output_step):
                if step == 0:
                    decoder_input = BOS
                else:
                    decoder_input = decoder_output.max(1)[-1].resize(1, 1)
                    decoder_input = self.embedding(decoder_input)
                attention_weights = self.attention('dot', z, first_encoder_output[:self.video_step])
                context = torch.bmm(attention_weights.transpose(1,2),
                                     first_encoder_output[:self.video_step].transpose(0,1))
                # batch, 1, hidden_size*2
                decoder_input_2 = torch.cat((decoder_input, first_encoder_output[self.video_step+step].unsqueeze(1), context),2).transpose(0,1)

                decoder_output, z = self.decoder(decoder_input_2, z)
                decoder_output = self.softmax(self.out(decoder_output[0]))
                output = decoder_output.max(1)[-1].resize(1, 1)
                word2ix = output.data[0,0]
                ix2word = index2word[int(word2ix)]
                if word2ix < 3:
                    break
                else:
                    pred.append(ix2word)

            

        return pred
    
    def validation(self, video_seq, cap_seq, valid_size):
        loss = 0
        padding_encoder = Variable(torch.zeros(self.output_step, valid_size, 512)).cuda();
        padding_decoder = Variable(torch.zeros(self.video_step, valid_size, self.hidden_size+self.embedding_size)).cuda();
        BOS = [0] * valid_size
        BOS = Variable(torch.LongTensor([BOS])).resize(valid_size, 1).cuda()
        BOS = self.embedding(BOS)
        
        video_seq = F.selu(self.fc1(video_seq))

        encoder_input = torch.cat((video_seq, padding_encoder), 0)
        
        encoder_output, encoder_hidden = self.encoder(encoder_input)
        
        first_decoder_input = torch.cat((padding_decoder, encoder_output[:self.video_step,:,:]),2)
        first_encoder_output, decoder_hidden = self.decoder(first_decoder_input)
        z = decoder_hidden
        
        for step in range(self.output_step):
            if step == 0:
                decoder_input = BOS
            else:
                decoder_input = decoder_output.max(1)[-1].resize(valid_size, 1)
                decoder_input = self.embedding(decoder_input)
            attention_weights = self.attention('dot', z, encoder_output[:self.video_step])
            context = torch.bmm(attention_weights.transpose(1,2),
                                 encoder_output[:self.video_step].transpose(0,1))
            # batch, 1, hidden_size*2
            decoder_input_2 = torch.cat((decoder_input, encoder_output[self.video_step+step].unsqueeze(1), context),2).transpose(0,1)
            
            decoder_output, z = self.decoder(decoder_input_2, z)
            decoder_output = self.softmax(self.out(decoder_output[0]))
            loss += F.nll_loss(decoder_output, cap_seq[:,step])
        
        return loss





# In[13]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inverse_sigmoid(k, i):
    return k / (k + math.exp(i / k))


# In[14]:


class Attention(nn.Module):
    def __init__(self, batch_size, hidden_size, dropout=0.3):
        super(Attention, self).__init__()
        self.Attention = nn.Linear(hidden_size*2, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        
    def forward(self, mode, hidden, encoder_outputs):
        if mode == "nn":
            # (128, 80, 256)
            dup_hidden = hidden.transpose(0,1).expand(hidden.size()[1], 80, self.hidden_size)
            attn_output = self.Attention(torch.cat((encoder_outputs.transpose(0,1), dup_hidden), 2))
        elif mode == "dot":
            # (128, 80, 256) * (128, 256, 1) = (128, 80, 1)
            attention_output = torch.bmm(encoder_outputs.transpose(0,1), hidden.transpose(0,1).transpose(1,2))
        attention_output = F.tanh(attention_output)
        attention_weights = F.softmax(attention_output, dim=1)
        return attention_weights

validation_frames = []
validation_target = []

def create_validate():

    validation_label = pd.read_json(train_label_file).set_index('id')

    for index, row in validation_label[train_size:].iterrows():
        # tokenize caption
        for caption in np.unique(np.array(row['caption'])):
            new_caption = []
            for word in caption.split():
                word = word.strip(string.punctuation).lower()
                if word in video_caption_training_dataset.vocab.vocabulary:
                    new_caption.append(word)
                else:
                    new_caption.append("<UNK>")
            if (len(new_caption) + 1) > MAXLENGTHOFSENTENCE:
                continue
            # Number of padding to be appended
            cap_len = MAXLENGTHOFSENTENCE - (len(new_caption) + 1)
            new_caption += ["<EOS>"]
            new_caption += ["<PAD>"] * cap_len
            caption = [video_caption_training_dataset.vocab.word2index[word] for word in new_caption]
            validation_frames.append(np.load(f"{train_numpy_feat}{index}.npy"))
            validation_target.append(caption)
    validation_frames = Variable(torch.FloatTensor(validation_frames).transpose(0,1)).to(device)
    validation_target = Variable(torch.LongTensor(validation_target).view(-1, MAXLENGTHOFSENTENCE)).to(device)
    validation_frames.size(), validation_target.size()


def train():
    create_vocab()
    create_validate()
    hidden_size = 512
    batch_size = 64
    feature_size = 4096
    seq_len = 80
    iter_size = data_size // batch_size
    dataloader = DataLoader(video_caption_training_dataset, batch_size=batch_size, shuffle=True)
    s2vt = Seq2SeqVideoModel(feature_size,VOCAB_SIZE,hidden_size,seq_len,MAXLENGTHOFSENTENCE,batch_size)
    optimizer = optim.Adam(s2vt.parameters(), lr = 1e-3)
    s2vt.to(device)

    epoches = 50
    min_validation_loss = 99
    for epoch in range(epoches):
        s2vt.train()
        vid = 0
    #     teacher_forcing_ratio = inverse_sigmoid(10, epoch)
        teacher_forcing_ratio = 0.06
        epoch_losses = 0
        for i, batch_data in enumerate(dataloader):
            if i == iter_size:
                break
            optimizer.zero_grad()
            target = Variable(batch_data['onehot']).to(device)
            video_seq = Variable(batch_data['frame'].transpose(0, 1)).to(device)

            loss = s2vt(video_seq, target.max(2)[-1], teacher_forcing_ratio)


            epoch_losses += loss.item() / MAXLENGTHOFSENTENCE
            loss.backward()
            optimizer.step()
        ## validate
        s2vt.eval()
        validation_loss = 0

        for i in range(10):

            validation_loss += s2vt.validation(validation_frames[:,vid:vid+77,:], validation_target[vid:vid+77,:], 77).item() / MAXLENGTHOFSENTENCE
            vid += 77
        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            torch.save(s2vt.state_dict(), "s2vt")
        print("[Epoch %d] Loss: %f,  Validation Loss: %f"           % (epoch+1, epoch_losses/iter_size, validation_loss/10/77*64))


# In[22]:

def test():
    word2index = pickle.load(open("word2index.pickle", "rb"))
    index2word = pickle.load(open("index2word.pickle", "rb"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VOCAB_SIZE = len(word2index)

    test_id_file = DATA_DIR + 'testing_data/id.txt'
    print(test_id_file)
    hidden_size = 512
    batch_size = 64
    feature_size = 4096
    seq_len = 80
    s2vt = Seq2SeqVideoModel(feature_size,VOCAB_SIZE,hidden_size,seq_len,MAXLENGTHOFSENTENCE,batch_size)
    s2vt.load_state_dict(torch.load("s2vt"))
    s2vt.to(device)

    test_frames = {}
    test_label = pd.read_fwf(test_id_file, header=None)
    for index, row in test_label.iterrows():
        test_file = f"{test_numpy_feat}{row[0]}.npy"
        test_frames[row[0]] = torch.FloatTensor(np.load(test_file))

    s2vt.eval()
    predictions = []
    indices = []
    use_beam_search = True
    beam_size = 3
    for i, row in test_label.iterrows():
        video_input = Variable(test_frames[row[0]].view(-1, 1, feature_size)).cuda()
        pred = s2vt.testing(video_input, index2word, use_beam_search, beam_size)
        pred[0] = pred[0].title()
        pred = " ".join(pred)
        predictions.append(pred)
        indices.append(row[0])
        # print(row[0] + " / " + pred)

    with open(output_file, 'w') as result_file:
        for i in range(100):
            result_file.write(indices[i] + "," + predictions[i] + "\n")
test()
# In[23]:


# test_frames = {}
# test_label = pd.read_json(test_label_file).set_index('id')
# for index, row in test_label.iterrows():
#     test_frames[index] = torch.FloatTensor(np.load(f"{test_numpy_feat}{index}.npy"))

# # s2vt_model.load_state_dict(torch.load("model/s2vt"))

# s2vt.eval()
# predictions = []
# indices = []
# use_beam_search = True
# beam_size = 2
# for i, row in test_label.iterrows():
#     video_input = Variable(test_frames[i].view(-1, 1, feature_size)).to(device)
#     pred = s2vt.testing(video_input, video_caption_training_dataset.vocab.index2word, use_beam_search, beam_size)
#     pred[0] = pred[0].title()
#     pred = " ".join(pred)
#     predictions.append(pred)
#     indices.append(i)
#     print(i + " / " + pred)


# with open('result.txt', 'w') as result_file:
#     for i in range(100):
#         result_file.write(indices[i] + "," + predictions[i] + "\n")


# # In[24]:


# test_frames = {}
# test_label = pd.read_json(test_label_file).set_index('id')
# for index, row in test_label.iterrows():
#     test_frames[index] = torch.FloatTensor(np.load(f"{test_numpy_feat}{index}.npy"))

# # s2vt_model.load_state_dict(torch.load("model/s2vt"))

# s2vt.eval()
# predictions = []
# indices = []
# use_beam_search = False
# beam_size = 2
# for i, row in test_label.iterrows():
#     video_input = Variable(test_frames[i].view(-1, 1, feature_size)).to(device)
#     pred = s2vt.testing(video_input, video_caption_training_dataset.vocab.index2word, use_beam_search, beam_size)
#     pred[0] = pred[0].title()
#     pred = " ".join(pred)
#     predictions.append(pred)
#     indices.append(i)
#     print(i + " / " + pred)


# with open('result1.txt', 'w') as result_file:
#     for i in range(100):
#         result_file.write(indices[i] + "," + predictions[i] + "\n")

