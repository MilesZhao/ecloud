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
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense, Activation, Dropout
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from util import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, "initial leanring rate")
flags.DEFINE_integer('num_epochs', 100, "the number of epochs for training")
flags.DEFINE_integer('dim', 80, "reduced dim")
tf.flags.DEFINE_string("dat_type", "shear", "target val for predicting")
decay_rate = FLAGS.learning_rate/FLAGS.num_epochs

def normal_net():
    init = tf.keras.initializers.glorot_uniform(seed=123)
    kr = tf.keras.regularizers.l2(0.001)
    inputs = tf.keras.layers.Input(shape=(98, 96, 98,1))

    conv = tf.keras.layers.Conv3D(128, kernel_size=(7,7,7), strides=(7,7,7), \
        kernel_initializer=init, activation='relu')(inputs)

    conv = tf.keras.layers.Conv3D(512, kernel_size=(4,4,4), strides=(4,4,4), \
        kernel_initializer=init, activation='relu')(conv)

    conv = tf.keras.layers.Conv3D(512, kernel_size=(2,2,2), strides=(2,2,2), \
        kernel_initializer=init, activation='relu')(conv)
    
    conv = tf.keras.layers.MaxPooling3D(padding='same')(conv)
    conv = Dropout(rate=0.5, seed=123)(conv)
        
    # exit(conv)
    h = tf.keras.layers.Flatten()(conv)
    x = Dense(1000, kernel_initializer=init,kernel_regularizer=kr,bias_regularizer=kr, activation='relu')(h)
    x = Dropout(rate=0.5, seed=123)(x)

    x = Dense(128, kernel_initializer=init,kernel_regularizer=kr,bias_regularizer=kr, activation='relu')(x)
    x = Dropout(rate=0.5, seed=123)(x)
    x = Dense(128, kernel_initializer=init,kernel_regularizer=kr,bias_regularizer=kr, activation='relu')(x)
    x = Dropout(rate=0.5, seed=123)(x)
    x = Dense(128, kernel_initializer=init,kernel_regularizer=kr,bias_regularizer=kr, activation='relu')(x)
    x = Dropout(rate=0.5, seed=123)(x)

    x = Dense(64, kernel_initializer=init,kernel_regularizer=kr,bias_regularizer=kr, activation='relu')(x)
    x = Dropout(rate=0.5, seed=123)(x)
    x = Dense(64, kernel_initializer=init,kernel_regularizer=kr,bias_regularizer=kr, activation='relu')(x)
    x = Dropout(rate=0.5, seed=123)(x)

    x = Dense(32, kernel_initializer=init,kernel_regularizer=kr,bias_regularizer=kr, activation='relu')(x)
    x = Dropout(rate=0.5, seed=123)(x)
    x = Dense(1, kernel_initializer=init)(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


# model = normal_net()
X,y,_ = load_all(typ=FLAGS.dat_type,src='mix',shape=[98, 96, 98])
kf = KFold(n_splits=5, shuffle=True)
cnt = 1
r2_log, mae_log, rmse_log, rc_log = [], [], [], []
for trn_ids, tst_ids in kf.split(y):
    trn_X, trn_y = X[trn_ids], y[trn_ids]
    tst_X, tst_y = X[tst_ids], y[tst_ids]

    model = normal_net()
    sgd = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, momentum=0.9, decay=decay_rate, nesterov=True)
    model.compile(loss='mae', optimizer=sgd)

    def step_decay(epoch):
        initial_lrate = FLAGS.learning_rate
        drop = 0.5
        epochs_drop = 10
        lrate = initial_lrate * math.pow(drop, (1+epoch)//epochs_drop)
        return lrate
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    callbacks = [lrate]
    model.fit(trn_X, trn_y, epochs=FLAGS.num_epochs, batch_size=32, verbose=1, callbacks=callbacks)
    y_hat = model.predict(tst_X).reshape(-1)
    r21, mae1, rmse1, rc1=eval_metrics(tst_y, y_hat)

    r2_log.append([r21])
    mae_log.append([mae1])
    rmse_log.append([rmse1])
    rc_log.append([rc1])
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





exit()













































df = pd.read_csv('data/mix.csv')
sample_names = df.iloc[:,0].values
np.random.shuffle(sample_names)

kf = KFold(n_splits=5, shuffle=True)
cnt = 1
r2_log, mse_log, rmse_log, rc_log = [], [], [], []
for trn_ids, tst_ids in kf.split(sample_names):
    val_ids = trn_ids[:int(len(trn_ids)*0.1)]
    trn_ids = trn_ids[int(len(trn_ids)*0.1):]

    trn_names = sample_names[trn_ids]
    val_names = sample_names[val_ids]
    tst_names = sample_names[tst_ids] 
    trn_gen = DataGen(trn_names,batch_size=32)  
    val_gen = DataGen(val_names,batch_size=32)
    tst_gen = DataGen(tst_names,batch_size=32)
    tst_y = [tst_gen.targets[n] for n in tst_names]

    model = normal_net2d()
    adam = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate, decay=decay_rate)
    sgd = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, momentum=0.9, decay=decay_rate, nesterov=True)
    model.compile(loss='mae', optimizer=sgd)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="/tmp/saved{}.h5".format(cnt),
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    def step_decay(epoch):
        initial_lrate = 0.008
        drop = 0.5
        epochs_drop = 20.0
        lrate = initial_lrate * math.pow(drop, (1+epoch)//epochs_drop)
        return lrate
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    callbacks = [checkpoint, ]

    model.fit_generator(
        generator=trn_gen,
        validation_data=val_gen,
        use_multiprocessing=True,
        workers=8,
        verbose=1,
        epochs=num_epochs,
        callbacks=callbacks
        )

    y_hat = model.predict_generator(generator=tst_gen,use_multiprocessing=True,workers=8).reshape(-1)
    r21, mse1, rmse1, rc1=eval_metrics(tst_y, y_hat)

    tf.keras.backend.clear_session()
    best_model = tf.keras.models.load_model("/tmp/saved{}.h5".format(cnt))
    y_hat = best_model.predict_generator(generator=tst_gen,use_multiprocessing=True,workers=8).reshape(-1)
    r22, mse2, rmse2, rc2=eval_metrics(tst_y, y_hat)

    r2_log.append([r21, r22])
    mse_log.append([mse1, mse2])
    rmse_log.append([rmse1, rmse2])
    rc_log.append([rc1, rc2])
    print('fold {}: '.format(cnt))
    cnt += 1

final_r2 = np.array(r2_log).mean(axis=0)
final_mse = np.array(mse_log).mean(axis=0)
final_rmse = np.array(rmse_log).mean(axis=0)
final_rankcoef = np.array(rc_log).mean(axis=0)
print('step size: ', FLAGS.learning_rate)
print('epochs: ', num_epochs)
print('\n5 folds finished')
print('Final validation score, R2: {}'.format(final_r2[0]))
print('Final validation score, MSE: {}'.format(final_mse[0]))
print('Final validation score, RMSE: {}'.format(final_rmse[0]))
print('Final validation score, Ranking Coef: {} \n'.format(final_rankcoef[0]))

print('\nbest model')
print('Final validation score, R2: {}'.format(final_r2[1]))
print('Final validation score, MSE: {}'.format(final_mse[1]))
print('Final validation score, RMSE: {}'.format(final_rmse[1]))
print('Final validation score, Ranking Coef: {} \n'.format(final_rankcoef[1]))





































