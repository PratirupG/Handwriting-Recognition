try:
    import os 
    import cv2 
    import time 
    import tqdm
    import string
    import numpy as np
    import pandas as pa 
    from CONFIG import config
    from dataloader import dataLoader
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
    from tensorflow.keras.layers import Dense,Conv2D,LSTM,GRU,Bidirectional,BatchNormalization,Input,MaxPool2D,Lambda
except Exception:
    print('ERROR IN LOADING LIBRARY {}'.format(Exception))


class MODEL_ARCHITECTURE:
    def __init__(self):
        self.char_list = string.ascii_letters + string.digits

    def CNN_BiLSTM(self,max_label_len):

        # Model architecture 
        input_ = Input(shape=(32,128,1))
        # CNN 
        conv2d_1 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(input_)
        maxpool_2d_1 = MaxPool2D(pool_size=(2,2),strides=2)(conv2d_1)

        conv2d_2 = Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(maxpool_2d_1)
        maxpool_2d_2 = MaxPool2D(pool_size=(2,2),strides=2)(conv2d_2)

        conv2d_3 = Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same')(maxpool_2d_2)
        conv2d_4 = Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same')(conv2d_3)

        maxpool_2d_3 = MaxPool2D(pool_size=(2,1))(conv2d_4)

        conv2d_5 = Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same')(maxpool_2d_3)
        batch_norm_5 = BatchNormalization()(conv2d_5)

        conv2d_6 = Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv2d_6)

        maxpool_2d_4 = MaxPool2D(pool_size=(2,1))(batch_norm_6)

        conv2d_7 = Conv2D(filters=512,kernel_size=(2,2),activation='relu')(maxpool_2d_4)
 
        squeezed = Lambda(lambda x: K.squeeze(x,1))(conv2d_7)

        blstm1 = Bidirectional(LSTM(256,return_sequences=True,dropout=0.2))(squeezed)
        
        # = BatchNormalization()(blstm1)

        blstm2 = Bidirectional(LSTM(256,return_sequences=True,dropout=0.2))(blstm1)

        #blstm3 = Bidirectional(LSTM(256,return_sequences=True,dropout=0.2))(blstm2)

        outputs = Dense(len(self.char_list)+1,activation='softmax')(blstm2) #(31,63)
        cnn_lstm_ = Model(input_,outputs)


        # LSTM layer inputs
        labels = Input(name='the_labels',shape=[max_label_len],dtype='float32')
        input_length = Input(name='input_length',shape=[1],dtype='int64')
        label_length = Input(name='label_length',shape=[1],dtype='int64')

        loss_out = Lambda(self.CTC_LOSS,output_shape=(1,),name='ctc')([outputs,labels,input_length,label_length])

        
        training_model = Model(inputs=[input_,labels,input_length,label_length],outputs=loss_out)
        return training_model,cnn_lstm_

    def CNN_BiGRU(self,max_label_len):

        # Model architecture 
        input_ = Input(shape=(32,128,1))
        # CNN 
        conv2d_1 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(input_)
        maxpool_2d_1 = MaxPool2D(pool_size=(2,2),strides=2)(conv2d_1)

        conv2d_2 = Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(maxpool_2d_1)
        maxpool_2d_2 = MaxPool2D(pool_size=(2,2),strides=2)(conv2d_2)

        conv2d_3 = Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same')(maxpool_2d_2)
        conv2d_4 = Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same')(conv2d_3)

        maxpool_2d_3 = MaxPool2D(pool_size=(2,1))(conv2d_4)

        conv2d_5 = Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same')(maxpool_2d_3)
        batch_norm_5 = BatchNormalization()(conv2d_5)

        conv2d_6 = Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv2d_6)

        maxpool_2d_4 = MaxPool2D(pool_size=(2,1))(batch_norm_6)

        conv2d_7 = Conv2D(filters=512,kernel_size=(2,2),activation='relu')(maxpool_2d_4)
 
        squeezed = Lambda(lambda x: K.squeeze(x,1))(conv2d_7)

        bgru1 = Bidirectional(GRU(256,return_sequences=True,dropout=0.2))(squeezed)

        bgru2 = Bidirectional(GRU(256,return_sequences=True,dropout=0.2))(bgru1)

        outputs = Dense(len(self.char_list)+1,activation='softmax')(bgru2) #(31,63)

        cnn_gru_ = Model(input_,outputs)

        # LSTM layer inputs
        labels = Input(name='the_labels',shape=[max_label_len],dtype='float32')
        input_length = Input(name='input_length',shape=[1],dtype='int64')
        label_length = Input(name='label_length',shape=[1],dtype='int64')

        loss_out = Lambda(self.CTC_LOSS,output_shape=(1,),name='ctc')([outputs,labels,input_length,label_length])

        
        training_model = Model(inputs=[input_,labels,input_length,label_length],outputs=loss_out)
        return training_model,cnn_gru_

    def CTC_LOSS(self,args):
        y_pred,labels,input_length,label_length = args 
        return K.ctc_batch_cost(labels,y_pred,input_length,label_length)