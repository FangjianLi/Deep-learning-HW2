

import tensorflow as tf
import pandas as pd
import numpy as np

import time
from utils_data_processing_1 import get_data_1
from utils_data_processing_1 import get_captions_list_1
from utils_data_sampling_1 import get_batch_samples_1

import pickle
# import fire
from elapsedtimer import ElapsedTimer
#from pathlib import Path


## some hyperparameters


dim_image = 4096
dim_hidden = 512 #size of the hidden states
batch_size = 128
lstm_steps = 80
video_lstm_step = 80
caption_lstm_step = 20
train_text_path = "./captions/training_label.json"
train_feat_path = "./feature_dirs_training"



learning_rate = 1e-4
epochs = 100
frame_step = 80 # we need to check what is frame_step
model_path = None

## Get the data


train_data = get_data_1(train_text_path, train_feat_path)
captions = get_captions_list_1(train_data)

with open('./Processed_data/word2idx.pkl', 'rb') as word2idx_file:
    word2idx = pickle.load(word2idx_file)


with open('./Processed_data/idx2word.pkl', 'rb') as idx2word_file:
    idx2word = pickle.load(idx2word_file)

n_words = len(word2idx)
print(n_words)

with tf.variable_scope("RNN_model"):
    word_emb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='word_emb') # n_words haven't been defined yet

    lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
    lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
    encode_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_W') #those are the prepossessing the feature dimensions
    encode_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_b')

    word_emb_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_emb_W') #project it to the number of words
    word_emb_b = tf.Variable(tf.zeros([n_words]), name='word_emb_b')


    ## Give the placeholder

    video = tf.placeholder(tf.float32, [batch_size, video_lstm_step, dim_image])

    condition = tf.placeholder(tf.float32, shape=(), name='the_condition' )
    video_mask = tf.placeholder(tf.float32, [batch_size, video_lstm_step]) # we need to double check the vieo_mask

    caption = tf.placeholder(tf.int32, [batch_size, caption_lstm_step+1])
    caption_mask = tf.placeholder(tf.float32, [batch_size, caption_lstm_step+1]) # we need to check why +1, and how to gety the caption mask

    video_flat = tf.reshape(video, [-1, dim_image])
    image_emb = tf.nn.xw_plus_b( video_flat, encode_W, encode_b )
    image_emb = tf.reshape(image_emb, [batch_size, lstm_steps, dim_hidden])

    state1 = tf.zeros([batch_size, lstm1.state_size])
    state2 = tf.zeros([batch_size, lstm2.state_size])

    padding = tf.zeros([batch_size, dim_hidden]) # in the decoding stage, the padding can be condiered as the inputs to the LSTm

    probs = [] #lets check it later
    loss = 0.0 #lets check it later as well

    # the encoding stage:

    for i in range(0, video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = lstm1(image_emb[:,i,:], state1) #the lstm function is (input, output)

            with tf.variable_scope("LSTM2"):
                output2, state2 = lstm2(tf.concat([padding, output1], 1), state2)

    current_embed_2 = tf.nn.embedding_lookup(word_emb, caption[:, 0])

    for i in range(0, caption_lstm_step):


        current_embed_1 = tf.nn.embedding_lookup(word_emb, caption[:, i])

        current_embed = tf.cond(condition>0, lambda: current_embed_1, lambda: current_embed_2)
        #epoch_number==0 or np.random.rand(1)>epoch_number/epochs



        tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("LSTM1"):
            output1, state1 = lstm1(padding, state1)

        with tf.variable_scope("LSTM2"):
            output2, state2 = lstm2(tf.concat([current_embed, output1],1), state2)

        labels = tf.expand_dims(caption[:, i+1], 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)

        concated = tf.concat([indices, labels],1)
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, n_words]), 1.0, 0.0) #we can check it later, what is the
        logit_words = tf.nn.xw_plus_b(output2, word_emb_W, word_emb_b)


        max_prob_index = tf.argmax(logit_words, 1)
        probs.append(logit_words)

        current_embed_2 = tf.nn.embedding_lookup(word_emb, max_prob_index)
        # current_embed_2 = tf.expand_dims(current_embed_2, 0)

        # Computing the loss

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words,labels=onehot_labels)
        cross_entropy = cross_entropy * caption_mask[:,i] # we need to check how to get the caption_mask
        probs.append(logit_words)

        current_loss = tf.reduce_sum(cross_entropy)/batch_size
        loss = loss + current_loss

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    scope_RNN = tf.get_variable_scope().name

RNN_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_RNN)

print(RNN_variables)



sess = tf.InteractiveSession()

saver = tf.train.Saver(max_to_keep=100, write_version=1)
tf.global_variables_initializer().run()

loss_out = open('loss_record.txt', 'a+')
val_loss_total = []
for epoch in range(0, epochs):
    val_loss_epoch = []

    index = np.arange(len(train_data))

    train_data.reset_index()
    np.random.shuffle(index)
    train_data = train_data.loc[index]

    current_train_data = train_data.groupby(['video_path']).first().reset_index()

    for start in range(0, len(current_train_data)-batch_size, batch_size):
        start_time = time.time()
        current_feats, current_video_masks, current_caption_matrix, current_caption_masks, _, _ \
            = get_batch_samples_1(current_train_data, word2idx, start, batch_size, video_lstm_step, dim_image, caption_lstm_step)

        if epoch == 0 or np.random.rand(1)>epoch/epochs:
            check_condition = 1
        else:
            check_condition = -1

        probs_val = sess.run(probs, feed_dict={
            video: current_feats,
            caption: current_caption_matrix,
            condition: check_condition
        })



        _, loss_val = sess.run(
            [train_op, loss],
            feed_dict={
                video: current_feats,
                video_mask: current_video_masks,
                caption: current_caption_matrix,
                caption_mask: current_caption_masks,
                condition: check_condition
            })
        val_loss_epoch.append(loss_val)

        print('Batch starting index: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ',
              str((time.time() - start_time)), 'Truth_value?', check_condition)
        loss_out.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + 'Truth_value?' + str(check_condition) + '\n')

    val_loss_total.append(np.mean(val_loss_epoch))
    if (epoch+1) % 5 == 0:
        saver.save(sess, "./saved_models/model_{}.ckpt".format(epoch))

np.save("loss_total.npy", val_loss_total)

loss_out.close()