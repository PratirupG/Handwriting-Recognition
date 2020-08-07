from all_models import MODEL_ARCHITECTURE
from CONFIG import config
import numpy as np
import string
import tensorflow as tf
from dataloader import dataLoader
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from tensorflow.python.client import  device_lib
print(device_lib.list_local_devices())
sess = tf.compat.v1.Session(config= tf.compat.v1.ConfigProto(log_device_placement=True))

def TRAIN_MODEL(modelname):
    # Load all the data
    image,text,input_len,label_len,original_text,max_len = dataLoader().prepareData(iam=True,cvl=False)

    if modelname == 'lstm':
        model,model_arch = MODEL_ARCHITECTURE().CNN_BiLSTM(max_len)
    else:
        model,model_arch = MODEL_ARCHITECTURE().CNN_BiGRU(max_len)
    
    # Compile loaded model
    try:
        model.compile(loss={'ctc':lambda y_true,y_pred: y_pred},optimizer='adam')
    except:
        print('ERROR IN MODEL COMPILATION')

    # Create model checkpoint
    filepath = 'best_CNN_LSTM_model_final.hdf5'
    checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
    callback_list = [checkpoint]

    #Create train and validation data
    new_image,new_text,new_input_len,new_label_len,new_original_text =[],[],[],[],[]
    
    for index,len_ in enumerate(input_len):
        if label_len[index] == len_ and len_<=19 and label_len[index]!=0:
            new_image.append(image[index])
            new_text.append(text[index])
            new_input_len.append(31)
            new_label_len.append(label_len[index])        
            new_original_text.append(original_text[index])
    print('DATA LOADED SUCESSFULLY') 

    # Select 10% data for validation and 90% for training 
    total_train_len = int(abs((len(image) * 90) / 100))
    print("TOTAL LENGTH OD TRAINING DATA {}".format(total_train_len))
    
    train_image = np.array(new_image[:total_train_len])
    train_text = np.array(new_text[:total_train_len])
    train_input_len = np.array(new_input_len[:total_train_len])
    train_label_len = np.array(new_label_len[:total_train_len])
    #original_trainlabel = np.array(new_original_text[:total_train_len])

    valid_image = np.array(new_image[total_train_len+1:])
    valid_text = np.array(new_text[total_train_len+1:])
    valid_input_len = np.array(new_input_len[total_train_len+1:])
    valid_label_len = np.array(new_label_len[total_train_len+1:])
    #original_valid_label = np.array(new_original_text[total_train_len+1:])

    char_list = string.ascii_letters + string.digits

    train_padded_text = pad_sequences(train_text,maxlen=max_len,padding='post',value=len(char_list))
    valid_padded_text = pad_sequences(valid_text,maxlen=max_len,padding='post',value=len(char_list))


    # Train the model
    model.fit(x=[train_image,train_padded_text,train_input_len,train_label_len],y=np.zeros(len(train_image)),batch_size=config.BATCH_SIZE,epochs=config.EPOCHS,
          validation_data =([valid_image,valid_padded_text,valid_input_len,valid_label_len],[np.zeros(len(valid_image))]),
          verbose=1,callbacks=callback_list)

    # Save the trained_model in disk
    model.save_weights("best_CNN_LSTM_model_final_2.h5")

    model_json = model_arch.to_json()
    with open("CNN_LSTM_FINAL_2.json", "w") as json_file:
        json_file.write(model_json)
    
    print("MODEL SAVED SUCESSFULLY ")


TRAIN_MODEL('lstm')




