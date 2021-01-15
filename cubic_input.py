import tensorflow as tf
import numpy as np
import multiprocessing as mp

class CubicDataGen(object):
    """
    queue for fetching model data
    """
    def __init__(self, 
        files,
        dir_file= '/data/yong/proj_data/mp_80k_LTC/cloud_arr/',
        batch_size = 4,
        buffer_size = 8,
        reshape = [120,120,120],
        scaling = 'zeroone'
    ):
        """
        Create a new pipelining data processing
        receive a list of tfrecords file names. Using this data,
        this class will create the iterator of tensorflow dataset

        Args:
            files: a list of tfrecord file names
            batch_size: number of samples per batch
            buffer_size: number of samples used as buffer for tensorflow prefetch function
            reshape: array for reshaping the model input
        """
        self.filenames = [dir_file+f+'.tfrecords' for f in files]
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reshape = reshape
        self.scaling = scaling

    def _process_inputs(self, inputs, epsilon=1e-12):
        if self.scaling == 'MinMax':
            return tf.div(
                    tf.subtract(
                        inputs,
                        tf.reduce_min(inputs)
                        ),
                    tf.math.maximum(
                        tf.subtract(
                                    tf.reduce_max(inputs),
                                    tf.reduce_min(inputs)
                                    ),
                        epsilon
                    )
                )

        elif self.scaling == 'sigmoid':
            return tf.sigmoid(inputs)

        elif self.scaling == 'std':
            mu, sigma = tf.nn.moments(inputs,axes=[0])
            return tf.div(
                    tf.subtract(inputs, mu),
                    sigma
                )

        elif self.scaling == 'zeroone':
            return inputs

    def _parse_function(self,example_proto):
        features = {
            "cubic_raw": tf.FixedLenFeature([np.prod(self.reshape)],dtype=tf.float32),
            "vals": tf.FixedLenFeature([3],dtype=tf.float32),
            'shape': tf.FixedLenFeature([3],dtype=tf.float32),
            "name": tf.FixedLenFeature([], dtype=tf.string)
            }
        parsed_features = tf.parse_single_example(example_proto, features)
        inputs = parsed_features["cubic_raw"]
        inputs = self._process_inputs(inputs)
        inputs = tf.reshape(inputs,self.reshape)
        inputs = tf.expand_dims(inputs, -1)
        return  inputs, \
                parsed_features["vals"], \
                parsed_features["shape"], \
                parsed_features["name"]

    def tf_iterator(self):
        dataset = tf.data.TFRecordDataset(self.filenames,num_parallel_reads=mp.cpu_count())
        dataset = dataset.map(map_func=self._parse_function,\
            num_parallel_calls=mp.cpu_count())
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.buffer_size)
        iterator = dataset.make_initializable_iterator()
        return iterator


class CubicDataPool(object):
    """
    queue for fetching model data
    """
    def __init__(self, 
        files,
        dir_file= '/data/yong/proj_data/mp_80k_LTC/cloud_arr/',
        batch_size = 4,
        buffer_size = 8,
        reshape = [120,120,120],
        scaling = 'zeroone'
    ):
        """
        Create a new pipelining data processing
        receive a list of tfrecords file names. Using this data,
        this class will create the iterator of tensorflow dataset

        Args:
            files: a list of tfrecord file names
            batch_size: number of samples per batch
            buffer_size: number of samples used as buffer for tensorflow prefetch function
            reshape: array for reshaping the model input
        """
        self.filenames = [dir_file+f+'.tfrecords' for f in files]
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reshape = reshape
        self.scaling = scaling

    def _process_inputs(self, inputs, epsilon=1e-12):
        if self.scaling == 'MinMax':
            return tf.div(
                    tf.subtract(
                        inputs,
                        tf.reduce_min(inputs)
                        ),
                    tf.math.maximum(
                        tf.subtract(
                                    tf.reduce_max(inputs),
                                    tf.reduce_min(inputs)
                                    ),
                        epsilon
                    )
                )

        elif self.scaling == 'sigmoid':
            return tf.sigmoid(inputs)

        elif self.scaling == 'std':
            mu, sigma = tf.nn.moments(inputs,axes=[0])
            return tf.div(
                    tf.subtract(inputs, mu),
                    sigma
                )

        elif self.scaling == 'zeroone':
            return inputs

    def _parse_function(self,example_proto):
        features = {
            "cubic_raw": tf.FixedLenFeature([np.prod(self.reshape)],dtype=tf.float32),
            "vals": tf.FixedLenFeature([3],dtype=tf.float32),
            'shape': tf.FixedLenFeature([3],dtype=tf.float32),
            "name": tf.FixedLenFeature([], dtype=tf.string)
            }
        parsed_features = tf.parse_single_example(example_proto, features)
        inputs = parsed_features["cubic_raw"]
        inputs = self._process_inputs(inputs)
        inputs = tf.reshape(inputs,self.reshape)
        inputs = tf.expand_dims(inputs, -1)
        inputs = tf.expand_dims(inputs, 0)
        inputs = tf.layers.average_pooling3d(inputs,pool_size=[4,4,4],
                strides = [4,4,4])
        inputs = inputs[0]
        # exit(inputs)
        return  inputs, \
                parsed_features["vals"], \
                parsed_features["shape"], \
                parsed_features["name"]

    def tf_iterator(self):
        dataset = tf.data.TFRecordDataset(self.filenames,num_parallel_reads=mp.cpu_count())
        dataset = dataset.map(map_func=self._parse_function,\
            num_parallel_calls=mp.cpu_count())
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.buffer_size)
        iterator = dataset.make_initializable_iterator()
        return iterator























