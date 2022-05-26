# -*- coding: utf-8 -*-
"""

snippet_cust_grad.py demonstrates how approximate activation derivative and 
feedback weights can be defined using customized gradient in TensorFlow. 

"""
import tensorflor as tf

@tf.custom_gradient
def actFunction(v_, mu):
    '''
    Replaces ReLU activation function with customized gradient, 
    where mu is used as the approximate activation derivative
    
    Input: 
        v_: n_batch x n_unit, pre-activation (voltage) 
        mu: scalar, smeared activation derivative
    Output: 
        z_: n_batch x n_unit, firing rate
    '''
    z_ = tf.nn.relu(v_)

    def grad(dy):
        dv = dy * mu
        return [dv, tf.zeros_like(mu)]

    return z_, grad

@tf.custom_gradient
def custom_rec_Wab(activity, Wrec, Wab):
    '''
    Performs weighted summation of incoming activity with customized gradient, 
    where Wab is used as the approximate backpropagation weights
    
    Input:
        activity: n_batch x n_unit, firing rate of neurons
        Wrec: n_unit x n_unit x 1, recurrent weight matrix (row->pre, col->post)
        Wab: n_unit x n_unit, approximate weight matrix (row->pre, col->post)
    Output:
        out: n_batch x n_unit x 1, weighted summation of incoming activity
    '''
    out = tf.einsum('bi,ijk->bjk',activity, Wrec)

    def grad(dy): #(b,j,1)
        drec = tf.squeeze(tf.einsum('bjk,pjk->bpk',dy, tf.expand_dims(Wab,axis=-1)))  
        pre_activity = tf.expand_dims(activity, axis=-2) #(b,1,i)
        dWrec = tf.reduce_sum(dy * pre_activity, axis=0) #(b,j,1)*(b,1,i)->(j,i)
        dWrec = tf.expand_dims(tf.transpose(dWrec), axis=-1) # (i,j,1)
        return drec, dWrec, tf.zeros_like(Wab)
    
    return out, grad