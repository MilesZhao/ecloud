import random
random.seed(123)
import numpy as np
np.random.seed(123)
from tensorflow import set_random_seed
set_random_seed(123)
import pandas as pd
import tensorflow as tf
import pickle
import os, types
import shutil
import matplotlib as mpl
mpl.use('agg')
import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()
import math
from util import *
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense, Activation, Dropout
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
os.environ["CUDA_VISIBLE_DEVICES"]="0"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0005, "initial leanring rate")
flags.DEFINE_integer('num_epochs', 75, "the number of epochs for training")
flags.DEFINE_integer('dim', 40, "reduced dim")
tf.flags.DEFINE_string("dat_type", "shear", "target val for predicting")
decay_rate = FLAGS.learning_rate/FLAGS.num_epochs

def senet_module(inputs,inter_dim):
    init = tf.keras.initializers.glorot_uniform(123)
    conv = tf.keras.layers.Conv2D(inter_dim, kernel_size=(1,1),strides=(1,1),\
        padding='same',activation='relu',kernel_initializer=init)(inputs)

    squeeze = tf.keras.layers.GlobalAveragePooling2D()(conv)
    excitation = Dense(inter_dim//2,kernel_initializer=init, activation='relu')(squeeze)
    excitation = Dense(inter_dim,kernel_initializer=init, activation='sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape((1,1,inter_dim))(excitation)
    conv = tf.keras.layers.BatchNormalization()(conv)
    scale = tf.keras.layers.Multiply()([conv, excitation])
    return scale


def inception(x,dim=40):
    b0 = tf.keras.layers.Conv2D(dim,kernel_size=(1,1),strides=(1,1),
        padding='same',activation='relu')(x)

    b1 = tf.keras.layers.Conv2D(dim,kernel_size=(1,1),strides=(1,1),
        padding='same',activation='relu')(x)
    b1 = tf.keras.layers.Conv2D(dim,kernel_size=(3,3),strides=(1,1),
        padding='same',activation='relu')(b1)

    b2 = tf.keras.layers.Conv2D(dim,kernel_size=(1,1),strides=(1,1),
        padding='same',activation='relu')(x)
    b2 = tf.keras.layers.Conv2D(dim,kernel_size=(3,3),strides=(1,1),
        padding='same',activation='relu')(b2)
    b2 = tf.keras.layers.Conv2D(dim,kernel_size=(3,3),strides=(1,1),
        padding='same',activation='relu')(b2)

    b3 = tf.keras.layers.AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    b3 = tf.keras.layers.Conv2D(dim,kernel_size=(1,1),strides=(1,1),
        padding='same',activation='relu')(b3)

    conv = tf.keras.layers.Concatenate(axis=-1)([b0,b1,b2,b3])
    pool = tf.keras.layers.MaxPooling2D()(conv)
    pool = tf.keras.layers.BatchNormalization()(pool)
    return pool

def alexnet(inputs, dim):
    init = tf.keras.initializers.glorot_uniform(123)
    x = inception(inputs,10)
    x = inception(x,10)
    x = inception(x,10)
    
    conv = tf.keras.layers.Conv2D(128,kernel_size=(3,3),strides=(1,1),\
        padding='valid',activation='relu',kernel_initializer=init)(x)
    conv = tf.keras.layers.Conv2D(256,kernel_size=(3,3),strides=(1,1),\
        padding='valid',activation='relu',kernel_initializer=init)(conv)
    pool = tf.keras.layers.MaxPooling2D()(conv)
    pool = tf.keras.layers.BatchNormalization()(pool)
    # exit(pool)
    return tf.keras.layers.Flatten()(pool)

def split_branch(x):
    x1 = []
    x2 = []
    x3 = []
    for i, matrix in enumerate(x):
        matrix = np.squeeze(matrix)

        rot = np.rot90(matrix,axes=(0,1),k=2)
        rot = np.expand_dims(rot, axis=-1)
        rot = np.expand_dims(rot, axis=0)
        x1.append(rot)

        rot = np.rot90(matrix,axes=(0,2),k=2)
        rot = np.expand_dims(rot, axis=-1)
        rot = np.expand_dims(rot, axis=0)
        x2.append(rot)

        rot = np.rot90(matrix,axes=(1,2),k=2)
        rot = np.expand_dims(rot, axis=-1)
        rot = np.expand_dims(rot, axis=0)
        x3.append(rot)
    x1 = np.concatenate(x1, axis=0)
    x2 = np.concatenate(x2, axis=0)
    x3 = np.concatenate(x3, axis=0)
    return x1, x2, x3

def normal_net2d(dim = 24):
    kr = tf.keras.regularizers.l2(0.0)
    init = tf.keras.initializers.glorot_uniform(123)

    inputs_xyz = tf.keras.layers.Input(shape=(60, 60, 60, 1))
    r_inputs_xyz = tf.keras.layers.Lambda(lambda x:x[:,:,:,:,0])(inputs_xyz)
    xyz_view = senet_module(r_inputs_xyz,inter_dim=dim)
    h_xyz = alexnet(xyz_view,dim=dim)

    inputs_xzy = tf.keras.layers.Input(shape=(60, 60, 60, 1))
    r_inputs_xzy = tf.keras.layers.Lambda(lambda x:x[:,:,:,:,0])(inputs_xzy)
    xzy_view = senet_module(r_inputs_xzy,inter_dim=dim)
    h_xzy = alexnet(xzy_view,dim=dim)

    inputs_zyx = tf.keras.layers.Input(shape=(60, 60, 60, 1))
    r_inputs_zyx = tf.keras.layers.Lambda(lambda x:x[:,:,:,:,0])(inputs_zyx)
    zyx_view = senet_module(r_inputs_zyx,inter_dim=dim)
    h_zyx = alexnet(zyx_view,dim=dim)

    inputs_yxz = tf.keras.layers.Input(shape=(60, 60, 60, 1))
    r_inputs_yxz = tf.keras.layers.Lambda(lambda x:x[:,:,:,:,0])(inputs_yxz)
    yxz_view = senet_module(r_inputs_yxz,inter_dim=dim)
    h_yxz = alexnet(yxz_view,dim=dim)

    h_flat = tf.keras.layers.Concatenate(axis=1)([h_xyz, h_xzy, h_zyx, h_yxz])

    x = Dense(4096,kernel_initializer=init,activation='relu',kernel_regularizer=kr)(h_flat)
    x = Dense(4096,kernel_initializer=init,activation='relu',kernel_regularizer=kr)(x)
    x = Dense(32,kernel_initializer=init,activation='relu',kernel_regularizer=kr)(x)
    x = Dense(1, kernel_initializer=init)(x)
    return tf.keras.models.Model(inputs=[inputs_xyz,inputs_xzy,inputs_zyx,inputs_yxz], outputs=x)

# model = normal_net2d(dim=FLAGS.dim)
X1,y,names = load_all(typ=FLAGS.dat_type,src='fcc',shape=[60, 60, 60])


kf = KFold(n_splits=5, shuffle=True)
cnt = 1
r2_log, mae_log, rmse_log, rc_log = [], [], [], []
for trn_ids, tst_ids in kf.split(y):
    trn_X1, trn_y = X1[trn_ids], y[trn_ids]
    trn_X2,trn_X3,trn_X4 = split_branch(trn_X1)
    tst_X1, tst_y = X1[tst_ids], y[tst_ids]
    tst_X2,tst_X3,tst_X4 = split_branch(tst_X1)

    model = normal_net2d(dim=FLAGS.dim)
    sgd = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, momentum=0.9, decay=decay_rate, nesterov=True)
    adam = tf.keras.optimizers.Adam()
    rmsprop = tf.keras.optimizers.RMSprop()
    model.compile(loss='mae', optimizer=sgd)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="/tmp/saved_senet{}.h5".format(cnt),
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    def step_decay(epoch):
        if epoch > FLAGS.num_epochs//2:
            initial_lrate = FLAGS.learning_rate
            drop = 0.9
            epochs_drop = 10
            lrate = initial_lrate * math.pow(drop, (1+epoch)//epochs_drop)
            return lrate
        else:
            return FLAGS.learning_rate 
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    callbacks = [lrate]
    model.fit([trn_X1,trn_X2,trn_X3,trn_X4], trn_y, 
        epochs=FLAGS.num_epochs, batch_size=32, verbose=1, 
        callbacks=callbacks, validation_split=0.2)
    y_hat = model.predict([tst_X1,tst_X2,tst_X3,tst_X4]).reshape(-1)
    r21, mae1, rmse1, rc1=eval_metrics(tst_y, y_hat)

    best_model = tf.keras.models.load_model("/tmp/saved_senet{}.h5".format(cnt))
    y_hat = best_model.predict(tst_X).reshape(-1)
    r22, mae2, rmse2, rc2=eval_metrics(tst_y, y_hat)


    r2_log.append([r21, r22])
    mae_log.append([mae1, mae2])
    rmse_log.append([rmse1, rmse2])
    rc_log.append([rc1, rc2])
    print('fold {}: '.format(cnt))
    cnt += 1
    tf.keras.backend.clear_session()

final_r2 = np.array(r2_log).mean(axis=0)
final_mae = np.array(mae_log).mean(axis=0)
final_rmse = np.array(rmse_log).mean(axis=0)
final_rankcoef = np.array(rc_log).mean(axis=0)
print('\nDATA TYPE: {} STEP SIZE: {} EPOCH: {}'.format(FLAGS.dat_type, FLAGS.learning_rate, FLAGS.num_epochs))
print('5 folds finished')
print('Final validation score, R2: {}'.format(final_r2[0]))
print('Final validation score, MAE: {}'.format(final_mae[0]))
print('Final validation score, RMSE: {}'.format(final_rmse[0]))
print('Final validation score, Ranking Coef: {} \n'.format(final_rankcoef[0]))

print('\nbest model')
print('Final validation score, R2: {}'.format(final_r2[1]))
print('Final validation score, MSE: {}'.format(final_mae[1]))
print('Final validation score, RMSE: {}'.format(final_rmse[1]))
print('Final validation score, Ranking Coef: {} \n'.format(final_rankcoef[1]))

















































































