import tensorflow as tf
import numpy as np
import layers2 as L
from prep_data import knn_K
import matplotlib.pyplot as plt

CONFIG = {'num_classes': 10, 'batch_size_train': 32, 'batch_size_eval':4,\
 'knn_K': knn_K, 'num_epochs':200, 'size_train':3991, 'size_eval':908 }

TRAIN_FILE_LIST = 'mn10train.txt'
EVAL_FILE_LIST = 'mn10eval.txt'
CAT_DICT = {'bathtub':0,
            'bed':1,
            'chair':2,
            'desk':3,
            'dresser':4,
            'monitor':5,
            'night_stand':6,
            'sofa':7,
            'table':8,
            'toilet':9}

CATEGORIES = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']

class training_data(object):
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def plot(self):
        x_idxs = range(len(self.train_losses))
        plt.plot(x_idxs, self.train_losses, 'b')
        plt.plot(x_idxs, self.val_losses, 'r')
        plt.plot(x_idxs, self.val_accs, 'g')

class classifier_model(object):
    def __init__(self,input_tensor, labels, cfg):
        self.input_tensor = input_tensor
        self.labels = labels
        self.cfg = cfg

        outp = L.conv2(self.input_tensor, 128, size=[1,cfg['knn_K']], stride=(1,1))    # 2048, 1, 128        
        outp = L.conv2(outp, 128) # 2046, 1, 128
        outp = L.conv2(outp, 128) # 2044, 1, 128
        outp = L.mp(outp)   # 1022, 1, 128
        outp = L.conv2(outp, 256) # 1020, 1, 256
        outp = L.conv2(outp, 256) # 1018, 1, 256
        outp = L.conv2(outp, 256) # 1016, 1, 256
        outp = L.mp(outp)   # 508, 1, 256
        outp = L.conv2(outp, 512) # 506, 1, 512
        outp = L.conv2(outp, 512) # 504, 1, 512
        outp = L.mp(outp)   # 252, 1, 512
        outp = L.conv2(outp, 1024) # 250, 1, 1024
        outp = L.conv2(outp, 1024) # 248, 1, 1024
        outp = L.mp(outp)   # 124, 1, 1024
        outp = L.conv2(outp, 2048) # 122, 1, 2048
        outp = L.conv2(outp, 2048) # 120, 1, 2048
        outp = L.mp(outp) # 60, 1, 2048
        outp = L.conv2(outp, 2048)  # 58, 1, 2048
        outp = L.conv2(outp, 2048)  # 56, 1, 2048
        outp = L.mp(outp) # 28, 1, 2048
        outp = L.conv2(outp, 4096)  # 26, 1, 4096
        outp = L.conv2(outp, 4096)  # 24, 1, 4096
        outp = L.mp(outp) # 12, 1, 4096
        outp = L.conv2(outp, 4096)  # 10, 1, 4096
        outp = L.conv2(outp, 4096)  # 8, 1, 4096

        outp = tf.layers.Flatten()(outp)
        outp = tf.layers.dense(outp, 1024, activation=tf.nn.relu)
        self.logits = tf.layers.dense(outp, self.cfg['num_classes'])

        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        self.preds = tf.argmax(input = self.logits,axis = 1)
        # self.accuracy = tf.metrics.accuracy(labels=self.labels, predictions=self.preds)

def train_sample_gen(batch_size=32):
    with open(TRAIN_FILE_LIST,'r') as f:
        trainFiles = np.array(f.read().split('\n')[:-1])
    num_files = trainFiles.shape[0]
    trainFiles = trainFiles[np.random.permutation(num_files)]

    count = -1
    while(1):
        inp_tensor = np.empty((batch_size,2048,knn_K,3))
        labels = np.zeros(batch_size)
        for i in range(batch_size):
            count += 1
            if count >= num_files:
                count = 0
                trainFiles = trainFiles[np.random.permutation(num_files)]

            # preparing the label file
            cat = trainFiles[count].split('/')[1]
            label = CAT_DICT[cat]

            # Input file
            fname = trainFiles[count]        
            inp_tensor[i,:,:,:] = np.load(fname)
            labels[i] = label
        yield inp_tensor.astype(np.float32), labels.astype(np.int64)

def eval_sample_gen(batch_size=32):
    with open(EVAL_FILE_LIST,'r') as f:
        evalFiles = np.array(f.read().split('\n')[:-1])
    num_files = evalFiles.shape[0]

    count = -1
    while(1):
        inp_tensor = np.empty((batch_size,2048,knn_K,3))
        labels = np.zeros(batch_size)
        for i in range(batch_size):
            count += 1
            if count >= num_files:
                count = 0

            # preparing the label file
            cat = evalFiles[count].split('/')[1]
            label = CAT_DICT[cat]

            # Input file
            fname = evalFiles[count]     
            # print(fname)   
            inp_tensor[i,:,:,:] = np.load(fname)
            labels[i] = label
        yield inp_tensor.astype(np.float32), labels.astype(np.int64)

def train(trn_data_generator, eval_data_generator, cfg):
    best_acc = 0
    best_acc_epoch = 0
    x = tf.placeholder(tf.float32, [None,2048,cfg['knn_K'],3])
    y = tf.placeholder(tf.int32, [None])
    model = classifier_model(x, y, cfg)
    train_dat = training_data()

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        print('Starting training')    

        sess.run(tf.global_variables_initializer())

        num_batches = cfg['size_train'] // cfg['batch_size_train']
        eval_num_batches = cfg['size_eval'] // cfg['batch_size_eval']      


        for epoch in range(cfg['num_epochs']):  
            train_losses = []  
            for i in range(num_batches):
                batch = next(trn_data_generator)
                feed_dict = {x:batch[0], y:batch[1]}
                _, loss = sess.run([optimizer, model.loss], feed_dict=feed_dict)
                train_losses.append(loss)
                print('Epoch', epoch, 'batch:', i, 'loss:', loss)
            train_loss_mean = np.mean(train_losses)
            train_dat.train_losses.append(train_loss_mean)
            print('Mean loss:',train_loss_mean)

            eval_preds = []
            eval_gts = []
            eval_losses = []

            # saver.restore(sess, 'saved_models/classifier_model.ckpt')
            for i in range(eval_num_batches):
                batch = next(eval_data_generator)
                feed_dict = {x:batch[0], y:batch[1]}
                loss, preds = sess.run([model.loss, model.preds], feed_dict=feed_dict)
                eval_preds.extend(preds)
                eval_gts.extend(batch[1])
                
                eval_losses.append(loss)
            eval_loss_mean = np.mean(eval_losses)
            eval_acc_mean = (np.where(np.array(eval_gts) == np.array(eval_preds))[0]).shape[0]/cfg['size_eval']

            train_dat.val_losses.append(eval_loss_mean)
            train_dat.val_accs.append(eval_acc_mean)
            print('Accuracy:', eval_acc_mean, 'loss:', eval_loss_mean)   

            train_dat.plot()
            plt.savefig('progress.png')
            if eval_acc_mean > best_acc:
                print('Improved accuracy. Saving ... ')
                best_acc = eval_acc_mean
                best_acc_epoch = epoch
                saver.save(sess, 'saved_models/classifier_model.ckpt')
            print('Best acc', best_acc, 'at epoch', best_acc_epoch)

    return train_dat
           

if __name__ == '__main__':
    dat = train(train_sample_gen(CONFIG['batch_size_train']), eval_sample_gen(CONFIG['batch_size_eval']), CONFIG)