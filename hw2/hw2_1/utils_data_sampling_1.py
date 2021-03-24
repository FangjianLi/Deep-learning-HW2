import pandas as pd
import os
import numpy as np

from utils_data_processing_1 import get_captions_list_sampling_1
from keras.preprocessing import sequence
import pickle




def get_batch_samples_1(current_train_data, word2idx, start, batch_size, video_lstm_step, dim_image, caption_lstm_step):
    end = start + batch_size
    current_batch = current_train_data[start:end]
    current_videos = current_batch['video_path'].values

    current_feats = np.zeros((batch_size, video_lstm_step, dim_image))
    current_feats_vals = list(map(lambda vid: np.load(vid),current_videos))
    current_feats_vals = np.array(current_feats_vals)

    current_video_masks = np.zeros((batch_size, video_lstm_step)) # we need to double check it
    for ind, feat in enumerate(current_feats_vals):
        current_feats[ind][:len(current_feats_vals[ind])] = feat
        current_video_masks[ind][:len(current_feats_vals[ind])] = 1

    current_captions = get_captions_list_sampling_1(current_batch)


    for idx, each_cap in enumerate(current_captions):
        word = each_cap.lower().split(' ')
        if len(word) < caption_lstm_step:
            current_captions[idx] = current_captions[idx] + ' <eos>'
        else:
            new_word = ''
            for i in range(caption_lstm_step - 1):
                new_word = new_word + word[i] + ' '
            current_captions[idx] = new_word + '<eos>'

    current_captions_str = current_captions.copy()

    current_caption_ind = []
    for cap in current_captions:
        current_word_ind = []
        for word in cap.lower().split(' '):
            if word in word2idx:
                current_word_ind.append(word2idx[word])
            else:

                current_word_ind.append(word2idx['<unk>'])
        current_caption_ind.append(current_word_ind)

    current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=caption_lstm_step)
    # print(np.shape(current_caption_matrix))
    current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int) #expand the dimension
    # print(current_caption_matrix)
    current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
    nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix)))
    # (nonzeros)

    for ind, row in enumerate(current_caption_masks):
        row[:nonzeros[ind]] = 1

    return current_feats, current_video_masks, current_caption_matrix, current_caption_masks, current_captions_str, current_videos


if __name__ == '__main__':
    dim_image = 4096
    dim_hidden = 512  # size of the hidden states
    batch_size = 8
    lstm_steps = 80
    video_lstm_step = 80
    caption_lstm_step = 20
    start = 0

    train_data = pd.read_csv('./Processed_data/train.csv', sep=',')

    with open('./Processed_data/word2idx.pkl', 'rb') as word2idx_file:
        word2idx = pickle.load(word2idx_file)

    print(type(word2idx))

    with open('./Processed_data/idx2word.pkl', 'rb') as idx2word_file:
        idx2word = pickle.load(idx2word_file)

    # test_data = pd.read_csv('./Processed_data/test.csv', sep=',')

    # change the sequence
    index = np.arange(len(train_data))
    train_data.reset_index()
    print(len(train_data))
    train_data = train_data.loc[index]
    print(len(train_data))
    current_train_data = train_data.groupby(['video_path']).first().reset_index()
    print(len(current_train_data))

    _, _, _, _, caption_str, video_path = get_batch_samples_1(current_train_data, word2idx, start, batch_size, video_lstm_step, dim_image, caption_lstm_step)
    print(caption_str)

