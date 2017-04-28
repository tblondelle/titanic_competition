# -*- coding: utf-8 -*-
"""
Created on Thr Apr 27 15:04:38 2017

@author: thomas
"""

import tensorflow as tf
import numpy as np
import pandas as pd

from read_data import get_submit_data


from model import Model2 as Model # CHANGE MODEL HERE
MODEL_DIRECTORY = 'model2' # CHANGE MODEL HERE


with tf.Session() as sess:
    
    # Create model.
    model = Model()

    # Restore variables.
    ckpt = tf.train.get_checkpoint_state(MODEL_DIRECTORY)
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Fetch data.
    x_test = get_submit_data()

    feed_dict = {model.input_x: x_test}
    
    # Run network.
    scores = sess.run([model.scores], feed_dict)
    print(scores)
    
    # Create csv.
    s = pd.DataFrame(1-np.argmax(scores[0], axis=1), 
                     index=range(892, 1310),
                     columns=['Survived'])
    s.index.name = "PassengerId"
    s.to_csv('predictions2.csv')

    print(s.describe())
    print(s)
    
    
   
