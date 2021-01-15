import numpy as np
import pandas as pd
import tensorflow as tf
import pickle 
import os, types
import math
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
import multiprocessing as mp
from collections import Counter
from scipy import stats
from scipy.ndimage import zoom

def get_max_axis(names):
    arr = []
    dat = []
    for fname in names:
        f = open(root+"{}/ELFCAR".format(fname))
        res = f.readlines()
        f.close()
        line_forward = sum(list(map(int, res[6].replace("\n","").split())))
        line_forward = line_forward + 9
        size = list(map(int, res[line_forward].replace("\n","").split()))
        arr.append(size)
        # output_pooling(1,size)
        # print(fname, size)
        dat.append([fname, size[0], size[1], size[2]])
    # print(stats.mode(arr, axis=0))
    # print("median of arr : ", np.median(arr, axis=0))
    # dat = np.array(dat)
    # dat = pd.DataFrame(
    #     data = dat,
    #     columns=['name', 'l', 'w', 'h']
    # )
    # dat.to_csv('data/fcc.csv', index=False)
    return np.amax(arr, axis=0)

def fetch_file(fname):
    f = open(fname)
    res = f.readlines()
    f.close()
    line_forward = sum(list(map(int, res[6].replace("\n","").split())))
    line_forward = line_forward + 9
    size = list(map(int, res[line_forward].replace("\n","").split()))
    start, end = line_forward+1, line_forward+1 + math.ceil(np.prod(size)/5)
    value_str = ''.join(res[start:end]).replace("\n"," ").split()
    point_arr = list(map(np.float32, value_str))
    point_arr = np.array(point_arr).reshape(size)
    return point_arr, point_arr.shape

def zoom_3D_clound(shape=[],dirname=''):
    matrix, ori_shape = fetch_file(root+"{}/ELFCAR".format(dirname))
    zeros = zoom(matrix, zoom=(shape[0]/ori_shape[0],shape[1]/ori_shape[1],shape[2]/ori_shape[2]),\
        mode='nearest')
    # print(zeros.shape)
    tfrecords_filename = '/data/yong/proj_data/mp_80k_LTC/zoom_fcc_elf/{}.tfrecords'.format(dirname)
    with tf.python_io.TFRecordWriter(tfrecords_filename) as writer:
        example = tf.train.Example(features=tf.train.Features(
            feature={
            'cubic_raw':  tf.train.Feature(
                        float_list=tf.train.FloatList(value=zeros.reshape(-1))),
            'vals': tf.train.Feature(
                        float_list=tf.train.FloatList(value=targets[dirname])),
            'shape': tf.train.Feature(
                        float_list=tf.train.FloatList(value=ori_shape)),
            'name': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[dirname.encode('utf-8')]))

            }))
        writer.write(example.SerializeToString())



if __name__ == '__main__':
    print('begin to process')
    root = "/data/yong/proj_data/mp_80k_LTC/MIX_ELF/"
    pd_youngs = pd.read_csv('data/fcc.csv')
    names = pd_youngs.iloc[:,0].values
    print(names.shape)
    # names = [ "_".join(s.split("_")[::-1]) for s in names]
    bulk = pd_youngs.iloc[:,1].values
    shear = pd_youngs.iloc[:,2].values

    targets = {}
    for i, n in enumerate(names):
        targets[n] = [bulk[i],shear[i]]
    
    # shape = get_max_axis(names)
    # print(shape)
    # zoom_3D_clound([84, 84, 84],names[0])
    # exit()
    pool = mp.Pool(processes=10)
    for i in range(len(names)):
        pool.apply_async(zoom_3D_clound,
            args=(
                [84, 84, 84],
                names[i]
                )
        )
    pool.close()
    pool.join()


















































