import numpy as np
import math
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from cubic_input import CubicDataGen
from pymatgen.core.composition import Composition
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from scipy.ndimage import rotate 

def load_all(typ = 'bulk',src='fcc', shape=[84, 84, 84],scale=False):
    data_dir = '/home/yong'
    df = pd.read_csv('data/{}2170.csv'.format(src))
    sample_names = df.iloc[:,0].values

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            trn_data = CubicDataGen(sample_names,
                dir_file='{}/padding_{}_elf_bak_before_intra/'.format(data_dir, src),
                batch_size=32,
                buffer_size = 100,
                reshape=shape,
                scaling = "zeroone").tf_iterator()
            trn_next_element = trn_data.get_next()
            sess.run(trn_data.initializer)
            trn_X, trn_y, names = [], [], []
            while True:
                try:
                    input_x,targets,ori_shape,n = sess.run(trn_next_element)
                    if scale:
                        ori_shape = np.array(ori_shape)[:,0]
                        ori_shape = ori_shape/shape[0]
                        ori_shape =ori_shape.reshape((ori_shape.shape[0],1,1,1,1))
                        input_x = input_x * ori_shape

                    if typ == 'bulk':
                        input_y = targets[:,0]
                    elif typ == 'shear':
                        input_y = targets[:,1]
                    trn_X.append(input_x)
                    trn_y.append(input_y)
                    names.append([name.decode('utf-8') for name in n])
                except tf.errors.OutOfRangeError:
                    trn_X = np.concatenate(trn_X, axis=0)
                    trn_y = np.concatenate(trn_y, axis=0)
                    names = np.concatenate(names, axis=0)
                    break
    print("Data shape: {} Y data shape: {}".format(trn_X.shape, trn_y.shape))
    return trn_X, trn_y, names

def eval_metrics(t,p):
    mae = sum(map(lambda x: abs(x[0]-x[1]), zip(t, p)))/len(t)
    SSres = sum(map(lambda x: (x[0]-x[1])**2, zip(t, p)))
    SStot = sum([(x-np.mean(t))**2 for x in t])
    r2 = 1-(SSres/SStot)
    mse = round(SSres/len(t),3)
    rmse = math.sqrt(mse)
    return round(r2, 3), round(mae,3), round(rmse, 3), round(spearmanr(t,p)[0], 3)

























































































