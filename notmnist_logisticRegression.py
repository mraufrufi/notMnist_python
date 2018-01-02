from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
from PIL import Image, ImageDraw
from scipy.spatial.distance import cosine

#Class defination with all procedures 
class notmnist_logistic:

    def __init__(self):
        
        self.im_size = 28  
        self.pixel_depth = 255.0  
        self.valid_data =[]
        self.valid_labels=[]
        self.train_data=[]
        self.train_labels=[]
    
    #Read procedure use to read image one by one Reading image one by one 
    def read_image(self, target_folder,folder):
        """read image files"""
        #join root folder and image folder path
        folder = os.path.join(target_folder, folder) 
        im_files = os.listdir(folder)
        #Numpy array for image file dataset 
        dataset = np.ndarray(shape=(len(im_files), self.im_size, self.im_size),
                                dtype=np.float32)
        print(folder)
        num_images = 0
        #Read images one by one and store into dataset
        for image in im_files:
            image_file = os.path.join(folder, image)
            try:
                im_data = (ndimage.imread(image_file).astype(float) - 
                                self.pixel_depth / 2) / self.pixel_depth
                #if image size is not 28x28
                if im_data.shape != (self.im_size, self.im_size):
                    raise Exception('Not a Standered image size: %s' % str(im_data.shape))
                dataset[num_images, :, :] = im_data
                num_images = num_images + 1
            except IOError as e:
                print('image unreadabe:', image_file, ':', e)
        dataset = dataset[0:num_images, :, :]
        
        print('dataset shape:', dataset.shape)
        print('Mean:', np.mean(dataset))
        print('Standard deviation:', np.std(dataset))
        return dataset


    #creat array fo combing all immage into sigle dataset 
    def make_arrays(self,nb_rows, img_size):
        if nb_rows:
            dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
            labels = np.ndarray(nb_rows, dtype=np.int32)
        else:
            dataset, labels = None, None
        return dataset, labels
    # Function to creat data set of all taraing , validatio and test 
    def create_data(self,image_folder, train_size, valid_size=0):

        data_folders = os.listdir(image_folder)
        data_folders.sort()
        num_classes = 10 
        valid_d, valid_l= self.make_arrays(valid_size, self.im_size)
        train_d, train_l= self.make_arrays(train_size, self.im_size)
        vsize_per_class = valid_size // num_classes
        tsize_per_class = train_size // num_classes
            
        start_v, start_t = 0, 0
        end_v, end_t = vsize_per_class, tsize_per_class
        end_l = vsize_per_class+tsize_per_class
        for label, im_folder in enumerate(data_folders):       
            try:
                dataset = self.read_image(image_folder,im_folder)
                np.random.shuffle(dataset)
                if valid_d is not None:
                    valid_l = dataset[:vsize_per_class, :, :]
                    valid_d[start_v:end_v, :, :] = valid_l
                    valid_l[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                            
                train_letter = dataset[vsize_per_class:end_l, :, :]
                train_d[start_t:end_t, :, :] = train_letter
                train_l[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
            except Exception as e:
                print('Unable to process data from', ':', e)
                raise
            
        return valid_d, valid_l, train_d, train_l
    
    #Call creat procedure and save the data set in to notMNIST.picke binary file 
    def save_data(self):

        train_root='notMNIST_large'
        test_root='notMNIST_small'
        train_size = 200000
        valid_size = 10000
        test_size = 10000
        pickle_file = 'notMNIST.pickle'

        self.valid_data, self.valid_labels, self.train_data, self.train_labels = self.create_data(train_root ,train_size,valid_size)
        _, _, self.test_data, self.test_labels = self.create_data(test_root, test_size)
        #combining the train , validation and test data in to singel pickle file
        try:
            f = open(pickle_file, 'wb')
            save = {
                'train_data': self.train_data,
                'train_labels': self.train_labels,
                'valid_data': self.valid_data,
                'valid_labels': self.valid_labels,
                'test_data': self.test_data,
                'test_labels': self.test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)

    #Traing the notmnist dataset 
    def training(self):

        self.train_data, self.train_label = self.randomize(self.train_data, self.train_label)
        self.test_data, self.test_label = self.randomize(self.test_data, self.test_label)
        self.valid_data, self.valid_label = self.randomize(self.valid_data, self.valid_label)

        for train_examples_count in [50,100,1000,5000]:
            #Trainin by using logistic regression funtion provided by sklearn library 
            logit =LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000)
            small_train_data = self.train_data[:train_examples_count]
            small_train_lbl = self.train_label[:train_examples_count]

            #Converting each image into 1*784 array line mnist dataset
            X_train = small_train_data.reshape(len(small_train_data),len(small_train_data[0])*len(small_train_data[0]))
            logit.fit(X_train,small_train_lbl)
            print(logit.score(X_train,small_train_lbl))

        #Trainign predction and testing 
        X_test = self.test_data.reshape(len(self.test_data),len(self.test_data[0])*len(self.test_data[0]))
        pred_lst = [(logit.predict(row.reshape(1,-1)))[0] for row in X_test]

        #Simple accuracy and error percentage calculation
        accuracy = (sum(np.equal(pred_lst,self.test_label))*100)/10000
        error = 100 - accuracy
        print('Prediction Accuracy : ' ,accuracy , '%')
        print('Prediction Erry : ' , error, '%')
  
    #This prodecure is used weather date already exist, 
    #if existes then load the file Other wise call save procedure to creat new data set
    def check_data(self):

        data_file = 'notMNIST.pickle'
        if os.path.exists(data_file):
            print('%s already present - Loading Data from file' % data_file)
            notMNIST = open('notMNIST.pickle', 'rb')
            data = pickle.load(notMNIST)
            notMNIST.close()
            self.train_data = data['train_data']
            self.train_label = data['train_labels']
            self.valid_data = data['valid_data']
            self.valid_label = data['valid_labels']
            self.test_data = data['test_data']
            self.test_label = data['test_labels']
            del data
        else:
            self.save_data()
    #Shuffel the dataset 
    def randomize(self,dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

#Class creationg and procedure call
not_logistic  = notmnist_logistic()
not_logistic.check_data()
not_logistic.training()