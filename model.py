# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:19:46 2017

@author: thomas
"""

import tensorflow as tf


def layer(input_size, output_size, x, name="simple_layer"):
    """
    Create a simple layer: y = ReLU(xW + b) and add summaries.
    """
    with tf.variable_scope(name):
        W = tf.Variable(tf.constant(0.1, shape=[input_size, output_size]), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='b')

        y = tf.nn.relu(tf.nn.xw_plus_b(x, W, b), name='y')
        
        tf.summary.histogram('weight', W)
        tf.summary.histogram('biais', b)
        tf.summary.histogram('y', y)        
    return y
        

class SimpleModel():
    
    def __init__(self):
        
        
        self.input_x = tf.placeholder(tf.float32, [None, 7], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        

        y = layer(7, 32, self.input_x, name="wxb_layer_1")
        y = layer(32, 32, y, name="wxb_layer_2")
        y = layer(32, 32, y, name="wxb_layer_3")
        y = layer(32, 2, y, name="wxb_layer_4") 
            
        with tf.variable_scope("softmax"):
            self.scores = tf.nn.softmax(y)
            tf.summary.histogram('scores', self.scores)

    
        with tf.variable_scope("results"):
            
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            
            correct_prediction = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
            
            
class Model2():
    
    def __init__(self):
        
        
        self.input_x = tf.placeholder(tf.float32, [None, 7], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        

        y = layer(7, 64, self.input_x, name="wxb_layer_1")
        y = layer(64, 64, y, name="wxb_layer_2")
        y = layer(64, 32, y, name="wxb_layer_3")
        y = layer(32, 32, y, name="wxb_layer_4")
        y = layer(32, 32, y, name="wxb_layer_5")
        y = layer(32, 2, y, name="wxb_layer_6") 
            
        with tf.variable_scope("softmax"):
            self.scores = tf.nn.softmax(y)
            tf.summary.histogram('scores', self.scores)

    
        with tf.variable_scope("results"):
            
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            
            correct_prediction = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        

            
            
            
            
            
            