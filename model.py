import tensorflow as tf
import numpy as np
import layers as L
from prep_data import knn_K
import matplotlib.pyplot as plt

CONFIG = {'num_classes': 10, 'batch_size_train': 32, 'batch_size_eval':4,\
 'knn_K': knn_K, 'num_epochs':80, 'size_train':3991, 'size_eval':908 }

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
    def __init__(self,input_tensor, labels, dropoutProb, cfg):
        self.input_tensor = input_tensor
        self.labels = labels
        self.cfg = cfg

        outp = L.conv2(self.input_tensor, dropoutProb, 128, size=[1,cfg['knn_K']], stride=(1,1))    # 2048, 1, 128

        # outp = L.conv2(self.input_tensor, 32, size=[1,3], stride=(1,1))    # 2048, 126, 32
        # outp = L.conv2(outp, 32, size=[1,3], stride=(1,1))    # 2048, 124, 32
        # outp = tf.layers.max_pooling2d(outp,[1,2],(1,2))    # 2048, 62, 32
        # outp = L.conv2(outp, 64, size=[1,3], stride=(1,1))    # 2048, 60, 64
        # outp = L.conv2(outp, 64, size=[1,3], stride=(1,1))    # 2048, 58, 64
        # outp = L.conv2(outp, 64, size=[1,3], stride=(1,1))    # 2048, 56, 64
        # outp = tf.layers.max_pooling2d(outp,[1,2],(1,2))    # 2048, 28, 64
        # outp = L.conv2(outp, 64, size=[1,3], stride=(1,1))    # 2048, 26, 64
        # outp = L.conv2(outp, 64, size=[1,3], stride=(1,1))    # 2048, 24, 64
        # outp = tf.layers.max_pooling2d(outp,[1,2],(1,2))    # 2048, 12, 64
        # outp = L.conv2(outp, 128, size=[1,3], stride=(1,1))    # 2048, 10, 128
        # outp = L.conv2(outp, 128, size=[1,3], stride=(1,1))    # 2048, 8, 128
        # outp = L.conv2(outp, 128, size=[1,8], stride=(1,1))    # 2048, 1, 128

        outp = L.conv2(outp, dropoutProb, 128) # 1024, 1, 128
        outp = L.conv2(outp, dropoutProb, 256) # 512, 1, 256
        outp = L.conv2(outp, dropoutProb, 256) # 256, 1, 256
        outp = L.conv2(outp, dropoutProb, 512) # 128, 1, 512
        outp = L.conv2(outp, dropoutProb, 512) # 64, 1, 512
        outp = L.conv2(outp, dropoutProb, 512) # 32, 1, 512
        outp = L.conv2(outp, dropoutProb, 1024) # 16, 1, 1024
        outp = L.conv2(outp, dropoutProb, 1024) # 8, 1, 1024

        outp = tf.layers.Flatten()(outp)
        # outp = tf.nn.dropout(tf.layers.dense(outp, 250, activation=tf.nn.relu), dropoutProb)
        outp = tf.layers.dense(outp, 250, activation=tf.nn.relu)
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
    dropoutProb = tf.placeholder(tf.float32, [])
    model = classifier_model(x, y, dropoutProb, cfg)
    train_dat = training_data()

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        print('Starting training')
        # _, loss = sess.run([optimizer, model.loss], feed_dict=feed_dict)

        

        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())


        # if args.load_params:
        #     ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
        #     print('Restoring parameters from', ckpt_file)
        #     saver.restore(sess, ckpt_file)

        num_batches = cfg['size_train'] // cfg['batch_size_train']
        eval_num_batches = cfg['size_eval'] // cfg['batch_size_eval']
       
        # if args.val_size > 0:
        #     validation = True
        #     val_num_batches = data.validation.num_examples // args.batch_size
        # else:
        #     validation = False

        for epoch in range(cfg['num_epochs']):  
            train_losses = []  
            for i in range(num_batches):
                batch = next(trn_data_generator)
                feed_dict = {x:batch[0], y:batch[1], dropoutProb: 0.5}
                _, loss = sess.run([optimizer, model.loss], feed_dict=feed_dict)
                train_losses.append(loss)
                # train_dat.train_losses.append(loss)
                print('Epoch', epoch, 'batch:', i, 'loss:', loss)
            train_loss_mean = np.mean(train_losses)
            train_dat.train_losses.append(train_loss_mean)
            print('Mean loss:',train_loss_mean)

            eval_preds = []
            eval_gts = []
            eval_losses = []

            # # tf.reset_default_graph()  
            # imported_meta = tf.train.import_meta_graph('saved_models/classifier_model.ckpt.meta') 
            # imported_meta.restore(sess, 'saved_models/classifier_model.ckpt') 
            # saver.restore(sess, 'saved_models/classifier_model.ckpt')
            for i in range(eval_num_batches):
                batch = next(eval_data_generator)
                feed_dict = {x:batch[0], y:batch[1], dropoutProb: 1.0}
                loss, preds = sess.run([model.loss, model.preds], feed_dict=feed_dict)
                eval_preds.extend(preds)
                eval_gts.extend(batch[1])
                
                eval_losses.append(loss)
                # eval_accuracies.append(acc)
            eval_loss_mean = np.mean(eval_losses)
            # eval_acc_mean = np.mean(eval_accuracies)
            # print(len(eval_gts))
            # print(len(eval_preds))
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


            # # compute loss over validation data
            # if validation:
                

            #     # log progress to console
            #     print("Epoch %d, time = %ds, train accuracy = %.4f, validation accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean, val_acc_mean))
            # else:
            #     print("Epoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean))
            # sys.stdout.flush()

            # if (epoch + 1) % 10 == 0:
            #     ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
            #     saver.save(sess, ckpt_file)

        # ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
        # saver.save(sess, 'saved_models/classifier_model.ckpt')
    return train_dat


    # G = tf.Graph()
    # with G.as_default():
    #     input_data_tensor = tf.placeholder(tf.float32, [None,2048,knn_K,3])
    #     input_label_tensor = tf.placeholder(tf.int32, [None])
    #     model = build_classifier(input_data_tensor, input_label_tensor)
    #     optimizer = tf.train.AdamOptimizer(0.009)
    #     grads = optimizer.compute_gradients(model["loss"])
    #     grad_step = optimizer.apply_gradients(grads)
    #     init = tf.initialize_all_variables()


    # config_proto = tf.ConfigProto(allow_soft_placement=True)
    # sess = tf.Session(graph=G, config=config_proto)
    # sess.run(init)
    # tf.train.start_queue_runners(sess=sess)
    # with sess.as_default():

    #     # Start training loop
    #     for step in range(80000):
    #         batch_train = next(trn_data_generator)
    #         X_trn = np.array(batch_train[0])
    #         Y_trn = np.array(batch_train[1])

    #         # print([k for k in sorted(model.keys())])

    #         ops = [grad_step] + [model[k] for k in sorted(model.keys())]
    #         inputs = {input_data_tensor: X_trn, input_label_tensor: Y_trn}
    #         results = sess.run(ops, feed_dict=inputs)
    #         results = dict(zip(sorted(model.keys()), results[1:]))
    #         acc = np.where(results["preds"] == Y_trn)[0].shape[0]/32.0
    #         print("TRN step:%-5d loss:%s, acc:%s" % (step, results["loss"], acc))


    #         if (step % 100 == 0):
    #             print("-- running evaluation on eval split")
    #             loss_ = 0
    #             acc_ = 0
    #             val_steps = int(908/32)
    #             for i in range(val_steps):
    #                 vld_data = next(eval_data_generator)
    #                 X_vld = vld_data[0]
    #                 Y_vld = vld_data[1]
    #                 inputs = {input_data_tensor:X_vld, input_label_tensor:Y_vld}
    #                 args = [X_vld, Y_vld]
    #                 ops = [model[k] for k in sorted(model.keys())]
    #                 # ops = [loss, accuracy]
    #                 results = sess.run(ops, feed_dict=inputs)
    #                 results = dict(zip(sorted(model.keys()), results))
    #                 acc = np.where(results["preds"] == Y_vld)[0].shape[0]/32.0
    #                 loss_ += results["loss"]
    #                 acc_ += acc

                    
    #             print("VLD step:%-5d loss:%s, accuracy:%s" % (step, loss_/val_steps, acc_/val_steps))
           

if __name__ == '__main__':
    dat = train(train_sample_gen(CONFIG['batch_size_train']), eval_sample_gen(CONFIG['batch_size_eval']), CONFIG)
    # dat.plot(CONFIG)