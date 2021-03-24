
import tensorflow as tf
import numpy as np
import os
import sys




from utils_data_sampling_1 import get_batch_samples_1
from utils_data_processing_1 import get_data_1


## some hyperparameterss


dim_image = 4096
dim_hidden = 512 #size of the hidden states
batch_size = 1
lstm_steps = 80
video_lstm_step = 80
caption_lstm_step = 20
# test_text_path = "./captions/testing_label.json"
# test_feat_path = sys.argv[1] #"./feature_dirs_testing"
#caption_test_path = sys.argv[2]#'generated_video_caption.txt'


frame_step = 80 # we need to check what is frame_step
model_path = None
import pickle



def test_it(num, test_feat_path = "./feature_dirs_testing", caption_test_path = 'generated_video_caption.txt'):
    tf.reset_default_graph()
    # test_data = get_data_1(test_text_path, test_feat_path)
    with open('./Processed_data/word2idx.pkl', 'rb') as word2idx_file:
        word2idx = pickle.load(word2idx_file)



    with open('./Processed_data/idx2word.pkl', 'rb') as idx2word_file:
        idx2word = pickle.load(idx2word_file)

    print(type(idx2word))
    n_words = len(word2idx)

    print(n_words)
    # print(len(test_data))

    ## build the model (same as seq to seq model)

    with tf.variable_scope("RNN_model"):
        word_emb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='word_emb')

        lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        encode_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_W')
        encode_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_b')

        word_emb_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_emb_W')
        word_emb_b = tf.Variable(tf.zeros([n_words]), name='word_emb_b')
        video = tf.placeholder(tf.float32, [1, video_lstm_step, dim_image])
        video_mask = tf.placeholder(tf.float32, [1, video_lstm_step])

        video_flat = tf.reshape(video, [-1, dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, encode_W, encode_b)
        image_emb = tf.reshape(image_emb, [1, video_lstm_step, dim_hidden])

        state1 = tf.zeros([1, lstm1.state_size])
        state2 = tf.zeros([1, lstm2.state_size])
        padding = tf.zeros([1, dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = lstm1(image_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, caption_lstm_step):
            tf.get_variable_scope().reuse_variables()

            if i == 0:

                current_embed = tf.nn.embedding_lookup(word_emb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                output1, state1 = lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = lstm2(tf.concat([current_embed, output1], 1), state2)

            logit_words = tf.nn.xw_plus_b(output2, word_emb_W, word_emb_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)


            current_embed = tf.nn.embedding_lookup(word_emb, max_prob_index)
            current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)
        scope_RNN = tf.get_variable_scope().name

    RNN_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_RNN)
    print(RNN_variables)


    #batch_size = len(test_data)

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, "./saved_models/model_{}.ckpt".format(num))



        # f.truncate(0)
        #print(len(test_data))

    
    video_list = os.listdir(test_feat_path)
    f = open(caption_test_path, 'w+')
    for video_path_short in video_list:
        video_path = test_feat_path + "/" + video_path_short

        # current_feats, current_video_masks, current_caption_matrix, current_caption_masks, current_caption_str, current_video_path =\
            # get_batch_samples_1(test_data, word2idx, i, batch_size, video_lstm_step, dim_image, caption_lstm_step)

        current_feats = np.load(video_path)


        gen_word_idx = sess.run(generated_words, feed_dict={video:[current_feats], video_mask:np.ones([1,frame_step])})

        #print(np.shape(current_video_masks))
        #gen_words = idx2word[gen_word_idx]


        gen_words = [idx2word[key] for key in gen_word_idx]

        

        punct = np.argmax(np.array(gen_words) == '<eos>') + 1
        gen_words = gen_words[:punct]

        gen_sent = ' '.join(gen_words)
        gen_sent = gen_sent.replace('<bos> ', '')
        gen_sent = gen_sent.replace(' <eos>', '')
        #print(current_caption_matrix)
        #print(gen_word_idx)

        #print(current_caption_str)
        #print(gen_sent)

        # current_caption_str = current_caption_str[0]
        # current_caption_str = current_caption_str.replace('<bos> ', '')
        # current_caption_str = current_caption_str.replace(' <eos>', '')
        #current_video_path= current_video_path[0].replace('./feature_dirs_testing/', '')
        current_video_path = video_path_short.replace('.npy', '')





        f.write(current_video_path+ ',')
        f.write(gen_sent + '\n')
        # f.write(current_caption_str + '\n')
    # generate the data for the testing

    f.close()
    print("______Done_______")
    
    
if __name__ == '__main__':
    test_feat_path = sys.argv[1] 
    caption_test_path = sys.argv[2]
    if not os.path.exists(caption_test_path):
    	os.mknod(caption_test_path)
    test_it(59, test_feat_path, caption_test_path)
