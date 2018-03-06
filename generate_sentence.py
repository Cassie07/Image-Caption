# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU,Dense, Embedding, Merge, Flatten, RepeatVector, TimeDistributed, Concatenate
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image as Image
from keras.preprocessing import sequence as Sequence
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model, to_categorical
from collections import Counter
CUDA_VISIBLE_DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

WEIGHTS_PATH = "C:/Users/dingk/Desktop/cse498/assignment3/v19_gru/checkpoint/weights-improvement-0.9025-0002.hdf5"
WORDS_PATH = "C:/Users/dingk/Desktop/cse498/assignment3/v19_gru/words.txt"
IMAGE_FILE = "C:/Users/dingk/Desktop/cse498/assignment3/flickr30k_images/flickr30k_images/97406261.jpg"
SENTENCE_MAX_LENGTH = 100
EMBEDDING_SIZE = 256
IMAGE_SIZE = 224

class Generate_Caption(object):

    def __init__(self, pra_voc_size):
        self.voc_size = pra_voc_size
        self.model = self.create_model()
        self.model.load_weights(WEIGHTS_PATH)

    def create_model(self):
        base_model = VGG19(weights='imagenet', include_top=True)
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        for layer in base_model.layers[1:]:
            layer.trainable = False
        # add a fully connected layer on the top of our base model
        # and repeat it several times, so that it has the same shape as our language model
        image_model = Sequential()
        image_model.add(base_model)
        image_model.add(Dense(EMBEDDING_SIZE, activation='relu'))
        image_model.add(RepeatVector(SENTENCE_MAX_LENGTH))
        # we use an Embedding layer to generate a good representation for captions.
        language_model = Sequential()
        # language_model.add(Embedding(self.voc_size, EMBEDDING_SIZE, input_length=SENTENCE_MAX_LENGTH))
        language_model.add(GRU(128, input_shape=(SENTENCE_MAX_LENGTH, self.voc_size), return_sequences=True))
        language_model.add(TimeDistributed(Dense(128)))
        # after merging CNN feature (image) and embedded vector (caption), we feed them into a LSTM model
        # at its end, we use a fully connected layer with softmax activation to convert the output into probability
        model = Sequential()
        model.add(Merge([image_model, language_model], mode='concat'))
        # model.add(Concatenate([image_model, language_model]))
        model.add(LSTM(1000, return_sequences=True))
        # model.add(Dense(self.voc_size, activation='softmax', name='final_output'))
        model.add(TimeDistributed(Dense(self.voc_size, activation='softmax')))
        # draw the model and save it to a file.
        # plot_model(model, to_file='model.pdf', show_shapes=True)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def get_dictionary(self, pra_captions):
            '''
           Generate a dictionary for all words in our dataset.
           Return:
               words2index: word->index dictionary
               index2words: index->word dictionary
           '''
            with open(WORDS_PATH, 'r',encoding='utf-8',errors='ignore') as reader:
                    words = [x.strip() for x in reader.readlines()]
            #print(words)
            self.voc_size = len(words)
            words2index = dict((w, ind) for ind, w in enumerate(words, start=0))
            index2words = dict((ind, w) for ind, w in enumerate(words, start=0))
            return words2index, index2words

    def caption2index(self, pra_captions):
        words2index, index2words = self.get_dictionary(pra_captions)
        captions = [x.split(' ') for x in pra_captions]
        index_captions = [[words2index[w] for w in cap if w in words2index.keys()] for cap in captions]
        return index_captions

    def index2caption(self, pra_index):
        words2index, index2words = self.get_dictionary('')
        captions = [' '.join([index2words[w] for w in cap]) for cap in pra_index]
        return captions

    def convert2onehot(self, pra_caption):
        captions = np.zeros((len(pra_caption), self.voc_size))
        for ind, cap in enumerate(pra_caption, start=0):
            captions[ind, cap] = 1
        return np.array(captions)

    def generate(self, image_path):
        # image
        image= Image.img_to_array(Image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE,3)))
        image=image[np.newaxis, :]  # 3 dimension to 4 dimension
        sentence_list = []
        input_word_list = ['<SOS>']
        input_word_index_list=self.caption2index(input_word_list) # str list to index list: [447] (index list)
        print('*****input_word_index_list=',input_word_index_list)# list
        for i in range(100):
            input_word_index_list_pad = Sequence.pad_sequences(input_word_index_list, maxlen=SENTENCE_MAX_LENGTH,
                                                     padding='post')
            print('*****input_word_index_list',input_word_index_list_pad)
            print('*****input_word_index_list', input_word_index_list_pad.shape)
            for index, zip_word_index_list in enumerate(zip(input_word_index_list_pad),   # 转置（python3）
                                                                start=1):
                input_onehot = self.convert2onehot(zip_word_index_list[0])
                print(input_onehot.shape)
            print('******input_onehot',input_onehot)
            print('******input_onehot_shape',input_onehot.shape)
            predict = self.model.predict([image, input_onehot[np.newaxis, :]]) # one_hot:2 dimension to 3 dimension
            print('*****predict',predict)
            print('*****predict_shape', predict.shape)
            output_word_line = predict[:,i,:]  # 取出第i行(array)
            print('*****output_word_line',output_word_line)
            print('*****output_word_line', output_word_line.shape)
            output_word_list = output_word_line.tolist()  # array to list
            print('*****output_word_list',output_word_list)
            index = output_word_list[0].index(max(output_word_list[0]))  # get the index of the Pmax
            print('*****index', index)
            print('*****predict_max',max(output_word_list[0]))
            input_word_index_list[0].append(index) # 下一次的generate()的输入，add index into index list [447,498]
            print('*****predict_input_word_index_list',input_word_index_list)
            word = self.index2caption(input_word_index_list)  # index to words
            print('*****word',word)
            #sentence_list.append(word)  # add this the word into sentence_list
            print('input_index_list******', input_word_index_list)
            print('sentence_list*******', word)
            if index == 592:    # <EOS>的index
                break

        pass

caption = Generate_Caption(12503)
caption.generate(IMAGE_FILE)
