import numpy as np 
import pandas as pa 
from tqdm import tqdm
from cv2 import cv2
import string
import os 
import fnmatch

class dataLoader:
    def __init__(self):
        self.path = r'D:\words'
        self.word_path = r'D:\ascii\words.txt'
    
    def read_data(self):
        word_path = r'D:\ascii\words.txt'
        line = 'start'
        static_path = 'D:/words/'
        image_name_path = []
        text = []
        len_text = []

        with open(word_path,'r') as reader:

            while line != '':

                line = reader.readline()

                split_line = line.split()
                try:
                    image_name = split_line[0]
                    text_name = split_line[-1]
                except:
                    break

                if text_name in string.punctuation or text is None or text == ' ' or text == '':
                    continue
                split_image_name = image_name.split('-')

                path = static_path +  split_image_name[0] + '/' + split_image_name[0] + '-' + split_image_name[1] + '/' + image_name + '.png'


                image_name_path.append(path)
                text.append(text_name) 

                len_text.append(len(text_name))

        return image_name_path,text,len_text

    def prepareData(self,cvl=False,iam=False,cvl_iam=False):
        input_len = []
        text_list = []
        original_text = []
        label_len = []
        max_len = 0
        image = []

        if iam:
            
            print('LOADING IAM DATASET')
            image_name_path,text,len_text = self.read_data()

        elif cvl:

            print('LOADING CVL DATASET')
            path = r'D:\cvl-database-1-1\testset\words'
            image_name_path = [ ]
            text = [ ]
            len_text = [ ]
            
            for root,dirs,files in os.walk(path):
                for image_name in files:
                    image_name_path.append(os.path.join(root,image_name))
                    val = image_name.split('-')[-1].split('.')[0]  
                    text.append(val.split('.')[0])
                    len_text.append(len(val))
        else:
            ## IN DEVELOPMENT DONT CALL THIS LOOP 
            
            image_name_path,text,len_text = self.read_data()
            
            for root,dirs,files in os.walk(path):
                for image_name in files:
                    image_name_path.append(os.path.join(root,image_name))
                    val = image_name.split('-')[-1].split('.')[0] 
                    text.append(val.split('.')[0])
                    len_text.append(len(val))
    
            

        for image_name,text_word,text_len in tqdm(zip(image_name_path,text,len_text)):
            
            load_image = cv2.imread(image_name)
            try:
                img = cv2.cvtColor(load_image,cv2.COLOR_BGR2GRAY)
                img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=91,C=11)
            except:
                continue

            # Dialiate and eroding of the image to make the image look much clearer
            kernal = np.ones((5,5),np.uint8)
            #img = cv2.erode(img,kernal,1)
            #img = cv2.dilate(img,kernal,1)

            # Image size adjustment to make all the image of shape (128,32)(h*w)
            height,width = img.shape
                
            if ((height > 32) or (width > 128)):
                img = cv2.resize(img,(128,32),interpolation = cv2.INTER_AREA)
            else:
                if height < 32:
                    add_ones = np.ones((32-height,width)) * 255
                    try: 
                        img = np.concatenate((img,add_ones))
                    except:
                        continue

                if width < 128:
                    add_ones = np.ones((height,128-width)) * 255 
                    try:
                        img = np.concatenate((img,add_ones),axis=1)
                    except:
                        continue
            
            img = np.expand_dims(img,axis=2)
            
            # Encode text
            encode_text = self.__encode_string_to_numbers(text_word)
            
            # Len of text
            if text_len > max_len:
                max_len = text_len
            
            
            image.append(img)
            text_list.append(encode_text)
            original_text.append(text_word)
            input_len.append(len(encode_text))
            label_len.append(len(text_word))
            
        return image,text_list,input_len,label_len,original_text,max_len
    
    def __encode_string_to_numbers(self,word):
        char_list = string.ascii_letters + string.digits
        string_list = []
        try:
            for value,char in enumerate(word):
                try:
                    string_list.append(char_list.index(char))
                except:
                    print('WARNING ---- Punctuation in character {}'.format(word))
            return string_list
        except:
            print ('ERROR IN FOR LOOOP ')

