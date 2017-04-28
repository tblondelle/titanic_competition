# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:04:38 2017

@author: thomas
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import datetime
import pandas as pd

import read_data as rd
from model import Model2 as Model  ## CHANGE MODEL HERE.


BATCH_SIZE = 100
LEARNING_RATE = 0.001
MODEL_DIRECTORY = 'model2' ## CHANGE DIRECTORY HERE.
SAVE_EVERY = 500    
TRAIN_LOOPS = 25000


def train():
    
    with tf.Graph().as_default() as graph:

        # Create model.
        print('Creating model...')
        model = Model() ## <--

        # Define Training procedure.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(model.cross_entropy, global_step=global_step)

        # Checkpoint directory.
        checkpoint_path = MODEL_DIRECTORY + "/checkpoint.ckpt"
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        
        
    with tf.Session(graph=graph) as sess:

        # Initialize.
        print('Initializing...')
        sess.run(tf.global_variables_initializer())

        # Maybe restore model parameters.
        ckpt = tf.train.get_checkpoint_state(MODEL_DIRECTORY)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + '.index'):
            print("Restoring model parameters from %s." % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Fresh parameters for this model.")

        # Tensorboard.
        dir_summary = MODEL_DIRECTORY +'/summary/' + datetime.datetime.now().isoformat()
        train_writer = tf.summary.FileWriter(dir_summary, sess.graph)
        merged_summary = tf.summary.merge_all()
        
        def train_step(x_batch, y_batch):
            """
            A single training step.
            """
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch}

            summary, _, step, accuracy = sess.run(
                [merged_summary, train_op, global_step, model.accuracy],
                feed_dict)

            train_writer.add_summary(summary, step)
            time_str = datetime.datetime.now().isoformat()
            if step%100 == 0:
                print("{}: step {}, accuracy {}".format(time_str, step, accuracy))
                
                
        def test_step(x, y):
            """
            Test the network with data unseen by the network.
            """
            feed_dict = {
              model.input_x: x,
              model.input_y: y}

            accuracy = sess.run(model.accuracy, feed_dict)

            print("Test set: accuracy {}".format(accuracy))
            with open(MODEL_DIRECTORY + '/log_accuracy.csv', 'a') as f:
                f.write(str(accuracy) + '\n')
            mean_sofar = pd.read_csv(MODEL_DIRECTORY + "/log_accuracy.csv", sep='\n').mean()
            print("Mean accuracy so far: {}\n".format(mean_sofar))

        # Create the batch generator.
        batch_generator = rd.get_new_batch(BATCH_SIZE)
        
        # Training loops.
        for _ in range(TRAIN_LOOPS):
            train_x, train_y = next(batch_generator)
            train_step(train_x, train_y)
            
            current_step = tf.train.global_step(sess, global_step)
            if current_step % SAVE_EVERY == 0:
                path = saver.save(sess, checkpoint_path, global_step=current_step)
                print("Saved model checkpoint to {}".format(path))
                
                test_x, test_y = rd.get_test_data()
                test_step(test_x, test_y)
                
        

if __name__ == '__main__':
    train()
        
