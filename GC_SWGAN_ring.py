#########################################################################
# File Name: GC-SWGAN-ring.py
# Author: Zhijian Luo
# Created Time: 2025-01-29 wed 05:20:01
#########################################################################
#!/usr/bin/env python3
import os
import time
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Activation, Reshape, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import Input, Dropout, Concatenate, Dense, LeakyReLU, Flatten, Embedding,multiply
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model,to_categorical
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

#***********************************************************************                                                                                                                 
import seaborn as sns
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
##**********************************************************************

gpu_list = tf.config.experimental.list_physical_devices('GPU')
if gpu_list:
    try:   
        tf.config.experimental.set_visible_devices(gpu_list[0], 'GPU')
    except RuntimeError as e:
        print(e)
MODEL_NAME = 'Find_Ring_galaxy'

OUTPUT_PATH = os.path.join('outputs', MODEL_NAME)
TRAIN_LOGDIR = os.path.join("logs", "tensorflow", MODEL_NAME, 'train_data') # Sets up a log directory.
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

lossfile = open(OUTPUT_PATH+'/loss.dat','w')

#TARGET_IMG_SIZE = 64 # Scale images to this size
crop_rate = 0.75
IMG_WIDTH = 192      #Images w size
IMG_HEIGHT = 192     #Images H size
CHANNELS = 3
BATCH_SIZE = 64
NOISE_DIM = 100
LAMBDA = 10  # For gradient penalty
NCLASSES = 2 # number of class
#alpha = 10 #for classifier

#################
STEPs = 100000
#################
CURRENT_STEP = 0 # Epoch start from
SAVE_EVERY_N_STEP = 100 # Save checkpoint at every n epoch

LR = 1e-4
MIN_LR = 0.000001 # Minimum value of learning rate
DECAY_FACTOR=1.000004 # learning rate decay factor

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(TRAIN_LOGDIR) #

# Prepare dataset

class Dataset:
    def __init__(self):
        # To get the images and labels from file
        with h5py.File('./decals_trains.h5', 'r') as F:
            images = np.array(F['images'][:,32:224,32:224,:])
            labels = np.array(F['ans'])
            images = images.astype(np.float32)

        train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.2, random_state = 16)
        self.x_train, self.y_train, self.x_test, self.y_test =  images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

        self.num_labeled = self.y_train.shape[0] 
        with h5py.File('./decals_images.h5','r') as FF:
            images_unsup = np.array(FF['images'][:,32:224,32:224,:])

        #with h5py.File('/home/zjluo/AI/GZD/decals_test_org.h5','r') as FF:
        #    images_unsup2 = np.array(FF['images'][:,32:224,32:224,:])

        self.decals_images = np.concatenate([images,images_unsup],axis=0)
        self.num_unlabeled = self.decals_images.shape[0] 

        def preprocess_imgs(x):
            x = (x.astype(np.float32) - 127.5) / 127.5
            return x

        def preprocess_labels(y):
            return y.reshape(-1, 1)

        
        self.x_train = preprocess_imgs(self.x_train)
        self.decals_images = preprocess_imgs(self.decals_images)
        self.y_train = preprocess_labels(self.y_train)
        
        self.x_test = preprocess_imgs(self.x_test)
        self.y_test = preprocess_labels(self.y_test)
        # flip
        def flip_left_90(arr): 
            return np.flip(arr.transpose((0,2,1,3)), axis=1)

        def flip_180_with_flip_left_90(arr):  
            return flip_left_90(flip_left_90(arr))

        def flip_180_with_axis(arr):          
            return np.flip(np.flip(arr, axis=2), axis=1)

        def flip_right_90_with_left_90(arr):  
            return flip_left_90(flip_left_90(flip_left_90(arr)))

        def flip_right_90_with_axis_left_90(arr): 
            return flip_left_90(flip_180_with_axis(arr))

        def flip_right_90_with_left_90_axis(arr):  
            return flip_180_with_axis(flip_left_90(arr))

        images_ud = np.flip(self.x_train, axis = 1)
        labels_ud = self.y_train.copy()

        images_lr = np.flip(self.x_train, axis = 2)
        labels_lr = self.y_train.copy()

        images_90l = flip_left_90(self.x_train)
        labels_90l = self.y_train.copy()
        images_90r = flip_right_90_with_left_90_axis(self.x_train)
        labels_90r = self.y_train.copy()
        images_180 = flip_180_with_axis(self.x_train)
        labels_180 = self.y_train.copy()

        self.images_ext = np.concatenate([self.x_train,images_ud,images_lr,images_90l,images_90r,images_180],axis = 0)
        self.labels_ext = np.concatenate([self.y_train,labels_ud,labels_lr,labels_90l,labels_90r,labels_180],axis = 0)



    def batch_labeled(self, batch_size):
        
        idx = np.random.randint(0, 6*self.num_labeled, batch_size)
        imgs = self.images_ext[idx]
        labels = self.labels_ext[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        
        idx = np.random.randint(0,self.num_unlabeled, batch_size)
        imgs = self.decals_images[idx]
        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def test_set(self):
        return self.x_test, self.y_test
######################################################################
dataset = Dataset()
num_labeled = dataset.num_labeled
num_unlabeled = dataset.num_unlabeled
######################################################################
def SGAN_generator(input_z_shape=NOISE_DIM):
        
    input_z_layer = Input(shape=(input_z_shape,))

    model = Sequential()
    model.add(Dense(6*6*256,use_bias=False,input_dim = input_z_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((6,6,256)))
    model.add(Conv2DTranspose(256,(4,4),strides=(1,1),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(256,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(CHANNELS,(4,4),strides=(2,2),padding='same',activation="tanh",use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))



    return model

generator = SGAN_generator()

generator.summary()

#plot_model(generator,show_shapes=True,to_file='g_model.png',dpi=64)

def SGAN_discriminator(input_x_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)):

    model = Sequential()
    model.add(Conv2D(64,(4,4),strides=(1,1),padding='same',input_shape=input_x_shape,use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(64,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(128,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(128,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(256,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(256,(4,4),strides=(2,2),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(256,(4,4),strides=(1,1),padding='same',use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.15))

    return model

discriminator = SGAN_discriminator()
discriminator.summary()
#plot_model(discriminator,to_file='d_model.png',show_shapes=True,dpi=64)

def SGAN_discriminator_unsup(discriminator):
    model = Sequential()
    model.add(discriminator)
    model.add(Dense(1))
    return model
discriminator_unsup = SGAN_discriminator_unsup(discriminator)

def SGAN_discriminator_sup(discriminator,num_classes=NCLASSES):
    model = Sequential()
    model.add(discriminator)
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
discriminator_sup = SGAN_discriminator_sup(discriminator,num_classes=NCLASSES)


# Optimizers 
D_sup_optimizer = Adam(learning_rate=LR, beta_1=0.5)
D_unsup_optimizer = Adam(learning_rate=LR, beta_1=0.5)
G_optimizer = Adam(learning_rate=LR, beta_1=0.5)

def learning_rate_decay(current_lr, decay_factor=DECAY_FACTOR):
    '''
    Calculate new learning rate using decay factor
    '''
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr):
    '''
    Set new learning rate to optimizers
    '''
    K.set_value(D_sup_optimizer.lr, new_lr)
    K.set_value(D_unsup_optimizer.lr, new_lr)
    K.set_value(G_optimizer.lr, new_lr)

#Setup Checkpoints

checkpoint_path = os.path.join("checkpoints", "tensorflow", MODEL_NAME)

ckpt = tf.train.Checkpoint(generator=generator,
                           discriminator=discriminator,
                           discriminator_sup = discriminator_sup,
                           discriminator_unsup = discriminator_unsup,
                           G_optimizer=G_optimizer,
                           D_sup_optimizer=D_sup_optimizer,
                           D_unsup_optimizer = D_unsup_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    latest_step = int(ckpt_manager.latest_checkpoint.split('-')[1])
    #print(latest_step)
    CURRENT_STEP = latest_step * SAVE_EVERY_N_STEP
    LR = G_optimizer.learning_rate.numpy()
    print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_STEP))


########################################
def generate_and_save_images(model,step, test_input, figure_size=(12,12), subplot=(10,10), save=False, is_flatten=False):
    '''
    Generate images and plot it.
    '''
    predictions = model.predict(test_input)
    if is_flatten:
        predictions = predictions.reshape(-1, IMG_WIDTH, IMG_HEIGHT, CHANNELS).astype('float32')
    fig = plt.figure(figsize=figure_size)
    for i in range(predictions.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, 'image_at_epoch_{:04d}.png'.format(step)))
    plt.show()

########################################
def plot_real_and_save_images(images,start_img=0, figure_size=(12,12), subplot=(2,2), save=False, is_flatten=False):
    '''
    show real images and plot it.
    '''
    real_img = images[start_img:start_img+subplot[0]*subplot[1]]
    if is_flatten:
        real_img = real_img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, CHANNELS).astype('float32')
    fig = plt.figure(figsize=figure_size)
    for i in range(real_img.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        plt.imshow(real_img[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, 'real_image_begin_no_{:04d}.png'.format(start_img+1)))
    plt.show()    

num_examples_to_generate = 9

# We will reuse this seed overtime
sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])


#Define training step

@tf.function
def CWGAN_GP_train_d_unsup_step(unsup_image, batch_size, step):
    '''
        One discriminator training step
    '''
    print("retrace")
    noise = tf.random.normal([batch_size, NOISE_DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    with tf.GradientTape(persistent=True) as d_unsup_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator(noise, training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(unsup_image, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred  = discriminator_unsup(fake_image_mixed, training=True)

        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

        fake_pred = discriminator_unsup(fake_image, training=True)
        unsup_pred = discriminator_unsup(unsup_image, training=True)

        D_unsup_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(unsup_pred) + LAMBDA * gradient_penalty
    # Calculate the gradients for discriminator
    D_unsup_gradients = d_unsup_tape.gradient(D_unsup_loss,
                                              discriminator_unsup.trainable_variables)
    # Apply the gradients to the optimizer
    D_unsup_optimizer.apply_gradients(zip(D_unsup_gradients,
                                          discriminator_unsup.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('D_unsup_loss', tf.reduce_mean(D_unsup_loss), step=step)
    return D_unsup_loss


@tf.function
def CWGAN_GP_train_g_step(unsup_image, batch_size, step):
    '''
        One generator training step
    '''
    print("retrace")
    noise = tf.random.normal([batch_size, NOISE_DIM])
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        fake_image = generator(noise, training=True)
        fake_pred = discriminator_unsup(fake_image, training=True)
        G_loss  = -tf.reduce_mean(fake_pred)
        # Calculate the gradients for generator
    G_gradients = g_tape.gradient(G_loss,
                                      generator.trainable_variables)
    # Apply the gradients to the optimizer
    G_optimizer.apply_gradients(zip(G_gradients,
                                      generator.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('G_loss', G_loss, step=step)
    return G_loss

@tf.function
def CWGAN_GP_train_d_sup_step(sup_image,label_oh,batch_size, step):
    with tf.GradientTape(persistent=True) as d_sup_tape:
        label_pred = discriminator_sup(sup_image,training=True)
        D_sup_loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy(from_logits=False)(label_oh,label_pred))
    D_sup_gradients = d_sup_tape.gradient(D_sup_loss,
                                          discriminator_sup.trainable_variables)
    #tf.print("D_sup_loss:", D_sup_loss)
    D_sup_optimizer.apply_gradients(zip(D_sup_gradients,
                                        discriminator_sup.trainable_variables))
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('D_sup_loss', D_sup_loss, step=step)
    return D_sup_loss


####################################################################################

# Start training

current_learning_rate = LR
trace = True

for step in range(CURRENT_STEP+1, STEPs + 1):   
    start = time.time()
    print('Start of epoch %d' % (step,))
    # Using learning rate decay
    current_learning_rate = learning_rate_decay(current_learning_rate)
    print('current_learning_rate %f' % (current_learning_rate,))
    set_learning_rate(current_learning_rate)

    sup_image,label = dataset.batch_labeled(BATCH_SIZE)
    label_oh = to_categorical(label,num_classes = NCLASSES)

    unsup_image = dataset.batch_unlabeled(BATCH_SIZE)

    D_sup_loss = CWGAN_GP_train_d_sup_step(sup_image,label_oh,batch_size=tf.constant(BATCH_SIZE, dtype=tf.int64),step=tf.constant(step,dtype=tf.int64))

    D_unsup_loss = CWGAN_GP_train_d_unsup_step(unsup_image,batch_size=tf.constant(BATCH_SIZE, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))

    # Train generator

    G_loss = CWGAN_GP_train_g_step(unsup_image,batch_size= tf.constant(BATCH_SIZE, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))

    if step % 10 == 0:
        print ('.', end='')


    if step % SAVE_EVERY_N_STEP == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for step {} at {}'.format(step,
                                                            ckpt_save_path))

    print ('Time taken for step {} is {} sec\n'.format(step,
                                                       time.time()-start))

    print("D_sup_loss = ", D_sup_loss.numpy())
    print("D_unsup_loss = ", D_unsup_loss.numpy())
    print("G_loss = ", G_loss.numpy())
    lossfile.write("%10.5f %10.5f %10.5f\n" % (D_sup_loss,D_unsup_loss,G_loss))


step =STEPs #template !!!!!!!!!!!!

generate_and_save_images(generator,step, sample_noise, figure_size=(12,12), subplot=(3,3), save=False, is_flatten=False)

image_test,label_test = dataset.test_set()

print(image_test.shape)

test_pred = discriminator_sup.predict(image_test)

#for row in range(len(test_pred)):
y_pred = np.argmax(test_pred,axis=1)
print(test_pred.shape)
y_true = label_test


# caculate conf_matrix

conf_matrix = confusion_matrix(y_true, y_pred)
cm_normalized = conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:,np.newaxis]
# print conf_matrix
print("Confusion matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion matrix')
plt.show()
# Calculate precision, recall and F1 Score
report = classification_report(y_true, y_pred, output_dict=True)
#print(report.items())


print("Macro Average:")  
print(f"  Precision: {report['macro avg']['precision']:.4f}")
print(f"  Recall: {report['macro avg']['recall']:.4f}")
print(f"  F1 score: {report['macro avg']['f1-score']:.4f}")
print()

print("Weighted Average:")  
print(f"  Precision: {report['weighted avg']['precision']:.4f}")
print(f"  Recall: {report['weighted avg']['recall']:.4f}")
print(f"  F1 score: {report['weighted avg']['f1-score']:.4f}")

for label, metrics in report.items():
    #if isinstance(label, int):
        print(f"class {label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1-score']:.4f}")
        print()














