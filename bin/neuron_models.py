"""
Code modified for rate-based neurons (ReLU activation) and for activation derivative
    approximation (for nonlocal terms only) for ModProp via automatic differentiation
    
The overall ModProp framework proposed is "communicating the credit information 
via cell-type-specific neuromodulators and processing it at the receiving cells 
via pre-determined temporal filtering taps." 

Current approximations are proof of concept for the framework, and better approximations 
can be developed as a part of the future work 

Modified from https://github.com/IGITUGraz/LSNN-official
    with the following copyright message retained from the original code:

##
The Clear BSD License

Copyright (c) 2019 the LSNN team, institute for theoretical computer science, TU Graz
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of LSNN nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from distutils.version import LooseVersion
import datetime
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

if LooseVersion(tf.__version__) >= LooseVersion("1.11"):
    from tensorflow.python.ops.variables import Variable, RefVariable
else:
    print("Using tensorflow version older then 1.11 -> skipping RefVariable storing")
    from tensorflow.python.ops.variables import Variable

from rewiring_tools_NP2 import weight_sampler
from lsnn.toolbox.tensorflow_einsums.einsum_re_written import einsum_bi_ijk_to_bjk, einsum_bij_jk_to_bik, einsum_bi_bij_to_bj
from lsnn.toolbox.tensorflow_utils import tf_roll

from time import time

Cell = tf.contrib.rnn.BasicRNNCell

def placeholder_container_for_rnn_state(cell_state_size, dtype, batch_size, name='TupleStateHolder'):
    with tf.name_scope(name):
        default_dict = cell_state_size._asdict()
        placeholder_dict = OrderedDict({})
        for k, v in default_dict.items():
            if np.shape(v) == ():
                v = [v]
            shape = np.concatenate([[batch_size], v])
            placeholder_dict[k] = tf.placeholder(shape=shape, dtype=dtype, name=k)

        placeholder_tuple = cell_state_size.__class__(**placeholder_dict)
        return placeholder_tuple


def feed_dict_with_placeholder_container(dict_to_update, state_holder, state_value, batch_selection=None):
    if state_value is None:
        return dict_to_update

    assert state_holder.__class__ == state_value.__class__, 'Should have the same class, got {} and {}'.format(
        state_holder.__class__, state_value.__class__)

    for k, v in state_value._asdict().items():
        if batch_selection is None:
            dict_to_update.update({state_holder._asdict()[k]: v})
        else:
            dict_to_update.update({state_holder._asdict()[k]: v[batch_selection]})

    return dict_to_update

@tf.custom_gradient
def rateFunction_1(v_, dampening_factor):
    '''
    Not used

    '''
    z_ = tf.nn.relu(v_)

    def grad(dy):
        psi = tf.where(tf.greater(v_, 0.), tf.ones_like(v_), tf.zeros_like(v_))
        psi_av = tf.reduce_mean(psi, axis=-1, keepdims=True) * dampening_factor

        dv = dy * psi_av

        return [dv, tf.zeros_like(dampening_factor)]

    return z_, grad

@tf.custom_gradient
def rateFunction_5(v_, dampening_factor, psiT):
    '''
    Not used

    '''
    z_ = tf.nn.relu(v_)

    def grad(dy):
        # psi_av = tf.where(tf.greater(v_, 0.), tf.ones_like(v_), tf.zeros_like(v_))
        psi_av = psiT * dampening_factor # use a constant psi

        dv = dy * psi_av

        return [dv, tf.zeros_like(dampening_factor), tf.zeros_like(psiT)]

    return z_, grad

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
def MDGL_output(activity, Wout, Wab, hj, decay):
    # Auto diff for MDGL
    logits = einsum_bij_jk_to_bik(activity, Wout)
    def grad(dy):
        dWout = tf.einsum('bij,bik->jk', activity, dy)
        dEdz = tf.einsum('btk,pk->btp', dy, Wout) #btp
        decay_ = tf.expand_dims(tf.expand_dims(decay,axis=0),axis=0)
        aj = dEdz*hj*(1-decay_) # (btj)
        mod1 = tf.pad(einsum_bij_jk_to_bik(aj[:,1:], Wab), ((0, 0), (0, 1), (0, 0)), 'constant') #(btj,jp->btp)
        dz = dEdz + mod1
        return [dz, dWout, tf.zeros_like(Wab), tf.zeros_like(hj), tf.zeros_like(decay)]

    return logits, grad   

@tf.custom_gradient
def custom_rec(z, Wrec, Wback):
    out = einsum_bi_ijk_to_bjk(z, Wrec)
    def grad(dy):
        drec = tf.squeeze(tf.einsum('bjk,pjk->bpk',dy, Wback)) 
        pre_activity = tf.expand_dims(z, axis=-2) #(b,1,i)
        dWrec = tf.reduce_sum(dy * pre_activity, axis=0) #(b,1,i)*(b,j,1)->(j,i)
        dWrec = tf.expand_dims(tf.transpose(dWrec), axis=-1) # (i,j,1)
        return drec, dWrec, tf.zeros_like(Wback)     
    return out, grad


@tf.custom_gradient
def filt_v(v_ghost, i_t, kappa, decay):
    out = tf.zeros_like(v_ghost) # v = kappa * v + (1 - decay) * i_t
    def grad(dy):
        dv = dy * kappa # (b,j)*(,j)
        dit = dy * (1 - decay)
        return [dv, dit, tf.zeros_like(kappa), tf.zeros_like(decay)]        
    return out, grad
 


def weight_matrix_with_delay_dimension(w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.
    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return:
    """
    with tf.name_scope('WeightDelayer'):
        w_d_list = []
        for kd in range(n_delay):
            mask = tf.equal(d, kd)
            w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w))
            w_d_list.append(w_d)

        delay_axis = len(d.shape)
        WD = tf.stack(w_d_list, axis=delay_axis)

    return WD


# PSP on output layer
def exp_convolve(tensor, decay):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float32]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor

def exp_convolve2(tensor, decay):  # tensor shape (trial, time, neuron)
    '''
    Not used
    '''
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float32]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tensor_time_major[0]

        filtered_tensor = tf.scan(lambda a, x: a * decay + x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor


GruStateTuple = namedtuple('GruStateTuple', ('z'))

class Gru(Cell):
    '''
    Not used 
    '''
    def __init__(self, n_in, n_rec, dtype=tf.float32, rewiring_connectivity=-1,
                 in_neuron_sign=None, rec_neuron_sign=None, custom_mode=0, w_adj=None, task='MNIST', wsig=0.0):

        self.custom_mode = custom_mode

        # Parameters
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.rewiring_connectivity = rewiring_connectivity
        self.in_neuron_sign = in_neuron_sign
        self.rec_neuron_sign = rec_neuron_sign
        
        self.wsig = wsig
        
        if w_adj is not None:
            W_zg_adj = w_adj['W_zg_adj']
            W_zr_adj = w_adj['W_zr_adj']
            W_zi_adj = w_adj['W_zi_adj']
        else:
            W_zg_adj = None
            W_zr_adj = None
            W_zi_adj = None

        with tf.variable_scope('InputWeights'):

            # Input weights
            if (0 < rewiring_connectivity < 1):
                self.W_ig, _, _, _ = weight_sampler(n_in, n_rec, rewiring_connectivity, neuron_sign=in_neuron_sign)
            else:
                self.W_ig = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype, name="InputaWeight")
                
            if (0 < rewiring_connectivity < 1):
                self.W_ir, _, _, _ = weight_sampler(n_in, n_rec, rewiring_connectivity, neuron_sign=in_neuron_sign)
            else:
                self.W_ir = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype, name="InputrWeight")
                
            if (0 < rewiring_connectivity < 1):
                self.W_ii, _, _, _ = weight_sampler(n_in, n_rec, rewiring_connectivity, neuron_sign=in_neuron_sign)
            else:
                self.W_ii = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype, name="InputiWeight")
                
            self.w_in_val = self.W_ii # for saving results

        with tf.variable_scope('RecWeights'):
            
            recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            
            if 0 < rewiring_connectivity < 1:
                self.W_zg, _,_,_ = weight_sampler(n_rec, n_rec, rewiring_connectivity, neuron_sign=rec_neuron_sign, is_con_0=W_zg_adj)
            else:
                if rec_neuron_sign is not None or in_neuron_sign is not None:
                    raise NotImplementedError('Neuron sign requested but this is only implemented with rewiring')
                self.W_zg = Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype, name='RecurrentaWeight')
            self.W_zg_ = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.W_zg), self.W_zg)  # Disconnect autotapse
            
            if 0 < rewiring_connectivity < 1:
                self.W_zr, _,_,_ = weight_sampler(n_rec, n_rec, rewiring_connectivity, neuron_sign=rec_neuron_sign, is_con_0=W_zr_adj)
            else:
                if rec_neuron_sign is not None or in_neuron_sign is not None:
                    raise NotImplementedError('Neuron sign requested but this is only implemented with rewiring')
                self.W_zr = Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype, name='RecurrentrWeight')
            self.W_zr_ = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.W_zr), self.W_zr)  # Disconnect autotapse
            
            if 0 < rewiring_connectivity < 1:
                self.W_zi, _,_,_ = weight_sampler(n_rec, n_rec, rewiring_connectivity, neuron_sign=rec_neuron_sign, is_con_0=W_zi_adj)
            else:
                if rec_neuron_sign is not None or in_neuron_sign is not None:
                    raise NotImplementedError('Neuron sign requested but this is only implemented with rewiring')
                self.W_zi = Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype, name='RecurrentiWeight')
            self.W_zi_ = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.W_zi), self.W_zi)  # Disconnect autotapse
            
            self.w_rec_val = self.W_zi # for saving results
            
        with tf.variable_scope('RecBiases'):            
            self.b_i = tf.Variable(tf.zeros_like(self.W_zi[0]), name='rec_bias_i')
            if task=='MNIST':
                self.b_g = tf.Variable(-3.0*tf.ones_like(self.b_i), trainable=True, name='rec_bias_g')
            else:
                self.b_g = tf.Variable(-0.0*tf.ones_like(self.b_i), trainable=True, name='rec_bias_g')
            self.b_r = tf.Variable(tf.zeros_like(self.W_zi[0]), name='rec_bias_r')

    @property
    def state_size(self):
        return GruStateTuple(z=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return GruStateTuple(z=z0)

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        # This MDGL implementation is very inefficient, 
        # but takes advantage of automatic differentiation 
        add_noise = tf.random.normal(state.z.shape, mean=0.0, stddev=self.wsig)
        mult_noise = tf.random.normal(state.z.shape, mean=1.0, stddev=self.wsig)
        
        if self.custom_mode == 2: #MDGL
            inputs1 = inputs[:,:self.n_in]
            inputs2 = inputs[:,self.n_in:]
            z_stop = tf.stop_gradient(state.z)
            
            g_t = tf.sigmoid(tf.matmul(inputs1, self.W_ig) + tf.matmul(z_stop, self.W_zg_) + self.b_g ) # bi,ij->bj
            r_t = tf.sigmoid(tf.matmul(inputs1, self.W_ir) + tf.matmul(z_stop, self.W_zr_) + self.b_r )
            i_t = tf.tanh(tf.matmul(inputs1, self.W_ii) + r_t*tf.matmul(z_stop, self.W_zi_) + self.b_i ) 
            z_1 = ((1-g_t)*state.z + g_t * i_t)*mult_noise + add_noise
            
            g_t = tf.sigmoid(tf.matmul(inputs2, self.W_ig) + tf.matmul(z_1, self.W_zg_) + self.b_g ) # bi,ij->bj
            r_t = tf.sigmoid(tf.matmul(inputs2, self.W_ir) + tf.matmul(z_1, self.W_zr_) + self.b_r )
            i_t = tf.tanh(tf.matmul(inputs2, self.W_ii) + r_t*tf.matmul(z_1, self.W_zi_) + self.b_i ) 
            new_z = ((1-g_t)*z_1 + g_t * i_t)*mult_noise + add_noise
        else:         
            if self.custom_mode>0: 
                z_stop = tf.stop_gradient(state.z)
            else:
                z_stop = state.z 
            
            # Explore sparse w to speed up sims; note, z is sparse for spiking neurons
            g_t = tf.sigmoid(tf.matmul(inputs, self.W_ig) + tf.matmul(z_stop, self.W_zg_) + self.b_g ) # bi,ij->bj
            r_t = tf.sigmoid(tf.matmul(inputs, self.W_ir) + tf.matmul(z_stop, self.W_zr_) + self.b_r )
            i_t = tf.tanh(tf.matmul(inputs, self.W_ii) + r_t*tf.matmul(z_stop, self.W_zi_) + self.b_i )        
            new_z = ((1-g_t)*state.z + g_t * i_t)*mult_noise + add_noise
            
        hpsi = new_z
        if self.custom_mode == 2: 
            new_state = GruStateTuple(z=z_1) # progress one step at a time
        else: 
            new_state = GruStateTuple(z=new_z)
        return [new_z, hpsi], new_state


LIFStateTuple = namedtuple('LIFStateTuple', ('v', 'z', 'i_future_buffer', 'z_buffer'))


def tf_cell_to_savable_dict(cell, sess, supplement={}):
    """
    Usefull function to return a python/numpy object from of of the tensorflow cell object defined here.
    The idea is simply that varaibles and Tensors given as attributes of the object with be replaced by there numpy value evaluated on the current tensorflow session.
    :param cell: tensorflow cell object
    :param sess: tensorflow session
    :param supplement: some possible
    :return:
    """

    dict_to_save = {}
    dict_to_save['cell_type'] = str(cell.__class__)
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dict_to_save['time_stamp'] = time_stamp

    dict_to_save.update(supplement)

    tftypes = [Variable, Tensor]
    if LooseVersion(tf.__version__) >= LooseVersion("1.11"):
        tftypes.append(RefVariable)

    for k, v in cell.__dict__.items():
        if k == 'self':
            pass
        elif type(v) in tftypes:
            dict_to_save[k] = sess.run(v)
        elif type(v) in [bool, int, float, np.int64, np.ndarray]:
            dict_to_save[k] = v
        else:
            print('WARNING: attribute of key {} and value {} has type {}, recoding it as string.'.format(k, v, type(v)))
            dict_to_save[k] = str(v)

    return dict_to_save


class LIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=1, rewiring_connectivity=-1,
                 in_neuron_sign=None, rec_neuron_sign=None,
                 dampening_factor=0.3,
                 injected_noise_current=0.,
                 V0=1., custom_mode=0, w_adj=None, task='mnist'):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
        :param reset: method of resetting membrane potential after spike thr-> by fixed threshold amount, zero-> to zero
        """

        if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)
        tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self.dampening_factor = dampening_factor
        self.custom_mode = custom_mode

        # Parameters
        self.n_delay = n_delay
        self.n_refractory = n_refractory

        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype
        self.n_E = int(0.8 * n_rec) + 1
        self.n_I = n_rec - self.n_E

        self._num_units = self.n_rec

        self.tau = tf.Variable(tau, dtype=dtype, name="Tau", trainable=False)
        self._decay = tf.exp(-dt / tau)
        self.thr = tf.Variable(thr, dtype=dtype, name="Threshold", trainable=False)

        self.V0 = V0
        self.injected_noise_current = injected_noise_current

        self.rewiring_connectivity = rewiring_connectivity
        self.in_neuron_sign = in_neuron_sign
        self.rec_neuron_sign = rec_neuron_sign
        self.task = task
        
        if w_adj is not None:
            wrec_adj = w_adj['wrec_adj']
        else:
            wrec_adj = None

        with tf.variable_scope('InputWeights'):

            # Input weights
            if task=='seqPred':
                self.W_in = tf.constant(np.expand_dims(rd.randn(n_in, n_rec) / np.sqrt(n_in), axis=-1), dtype=dtype, name="WinConst")
            else: 
                if (0 < rewiring_connectivity < 1) and (n_in>2):
                    self.w_in_val, self.w_in_sign, self.w_in_var, self.w_in_inicon = weight_sampler(n_in, n_rec, rewiring_connectivity,
                                                                                     neuron_sign=in_neuron_sign)
                else:
                    if (task=='mnist_row') or (task=='dlyXOR') or (task=='PGrate'):
                        self.w_in_var = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in/2), dtype=dtype, name="InputWeight")  
                    else:
                        self.w_in_var = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype, name="InputWeight")
                    self.w_in_val = self.w_in_var
    
                self.w_in_val = self.V0 * self.w_in_val
                self.w_in_delay = tf.Variable(rd.randint(self.n_delay, size=n_in * n_rec).reshape(n_in, n_rec),
                                              dtype=tf.int64, name="InDelays", trainable=False)
                self.W_in = weight_matrix_with_delay_dimension(self.w_in_val, self.w_in_delay, self.n_delay)

        with tf.variable_scope('RecWeights'):
            if 0 < rewiring_connectivity < 1:
                self.w_rec_val, self.w_rec_sign, self.w_rec_var, self.w_rec_inicon = weight_sampler(n_rec, n_rec,
                                                                                    rewiring_connectivity,
                                                                                    neuron_sign=rec_neuron_sign) 
            else:
                if rec_neuron_sign is not None or in_neuron_sign is not None:
                    raise NotImplementedError('Neuron sign requested but this is only implemented with rewiring')
                    
                if (task=='mnist_row') or (task=='dlyXOR') or (task=='PGrate'):
                    self.w_rec_var = Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec/2), dtype=dtype,
                                          name='RecurrentWeight')
                else:
                    self.w_rec_var = Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype,
                                          name='RecurrentWeight')
                self.w_rec_val = self.w_rec_var

            recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))

            self.w_rec_val = self.w_rec_val * self.V0
            self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),
                                      self.w_rec_val)  # Disconnect autotapse
            self.w_rec_delay = tf.Variable(rd.randint(self.n_delay, size=n_rec * n_rec).reshape(n_rec, n_rec),
                                           dtype=tf.int64, name="RecDelays", trainable=False)
            self.W_rec = weight_matrix_with_delay_dimension(self.w_rec_val, self.w_rec_delay, self.n_delay)

    @property
    def state_size(self):
        return LIFStateTuple(v=self.n_rec,
                             z=self.n_rec,
                             i_future_buffer=(self.n_rec, self.n_delay),
                             z_buffer=(self.n_rec, self.n_refractory))

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return LIFStateTuple(
            v=v0,
            z=z0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0
        )

    def __call__(self, inputs, state, scope=None, dtype=tf.float32): 
        if self.custom_mode>0: 
            z_stop = tf.stop_gradient(state.z)
        else:
            z_stop = state.z 
        i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
            z_stop, self.W_rec)

        new_v, new_z, psi = self.LIF_dynamic(
            v=state.v,
            z=state.z,
            z_buffer=state.z_buffer,
            i_future_buffer=i_future_buffer)

        new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
        new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

        new_state = LIFStateTuple(v=new_v,
                                  z=new_z,
                                  i_future_buffer=new_i_future_buffer,
                                  z_buffer=new_z_buffer)
        return [new_z, psi], new_state

    def LIF_dynamic(self, v, z, z_buffer, i_future_buffer, vghost, thr=None, decay=None, n_refractory=None, add_current=0.):
        """
        Function that generate the next spike and voltage tensor for given cell state.
        :param v
        :param z
        :param z_buffer:
        :param i_future_buffer:
        :param thr:
        :param decay:
        :param n_refractory:
        :param add_current:
        :return:
        """

        if self.injected_noise_current > 0:
            add_current = tf.random_normal(shape=z.shape, stddev=self.injected_noise_current)

        with tf.name_scope('LIFdynamic'):
            if thr is None: thr = self.thr
            if decay is None: decay = self._decay
            if n_refractory is None: n_refractory = self.n_refractory

            i_t = i_future_buffer[:, :, 0] + add_current

            # I_reset = z * thr * self.dt            
            new_v = decay * v + (1 - decay) * i_t #- I_reset
            if False: #self.custom_mode == 3:
                new_vghost = filt_v(vghost, i_t, self.kappa * (1-decay), decay)
            else:
                new_vghost = tf.zeros_like(v)

            # # Spike generation
            v_scaled = new_v - thr #(new_v - thr) / thr

            new_z = tf.nn.relu(v_scaled)

            new_z = new_z * 1 / self.dt
            psi = tf.gradients(new_z, new_v)[0]

            return new_v, new_z, psi, new_vghost


ALIFStateTuple = namedtuple('ALIFState', (
    'z',
    'v',
    'b',

    'i_future_buffer',
    'z_buffer', 'vghost'))


class ALIF(LIF):
    def __init__(self, n_in, n_rec, tau=20, thr=0.01,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
                 tau_adaptation=200., beta=1.6,
                 rewiring_connectivity=-1, dampening_factor=0.3,
                 in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
                 V0=1., custom_mode=0, trainable_adapt_mode=0, w_adj=None, task='mnist', MDGLpp_mode=-1):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
        :param tau_adaptation: adaptation time constant for the threshold voltage
        :param beta: amplitude of adpatation
        :param rewiring_connectivity: number of non-zero synapses in weight matrices (at initialization)
        :param in_neuron_sign: vector of +1, -1 to specify input neuron signs
        :param rec_neuron_sign: same of recurrent neurons
        :param injected_noise_current: amplitude of current noise
        :param V0: to choose voltage unit, specify the value of V0=1 Volt in the desired unit (example V0=1000 to set voltage in millivolts)
        """

        super(ALIF, self).__init__(n_in=n_in, n_rec=n_rec, tau=tau, thr=thr, dt=dt, n_refractory=n_refractory,
                                   dtype=dtype, n_delay=n_delay,
                                   rewiring_connectivity=rewiring_connectivity,
                                   dampening_factor=dampening_factor, in_neuron_sign=in_neuron_sign,
                                   rec_neuron_sign=rec_neuron_sign,
                                   injected_noise_current=injected_noise_current,
                                   V0=V0, custom_mode=custom_mode, w_adj=w_adj, task=task)

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")
        
        trainable_adapt = (trainable_adapt_mode>0)
        #self.train_logtau = (trainable_adapt_mode==2)
        if trainable_adapt_mode==2: # train log tau
            self.logtau = tf.Variable(np.log(tau_adaptation), dtype=dtype, name="logTauAdaptation", trainable=True)
            self.tau_adaptation = tf.exp(self.logtau)
        elif trainable_adapt_mode==3:
            self.exptau = tf.Variable(np.exp(-dt/tau_adaptation), dtype=dtype, name="expTauAdaptation", trainable=True)
            self.tau_adaptation = -dt/tf.log(self.exptau) 
        elif trainable_adapt_mode==4:
            self.exp50tau = tf.Variable(np.exp(-50*dt/tau_adaptation), dtype=dtype, name="exp50TauAdaptation", trainable=True)
            self.tau_adaptation = -50*dt/tf.log(self.exp50tau) 
        else: 
            self.tau_adaptation = tf.Variable(tau_adaptation, dtype=dtype, name="TauAdaptation", trainable=trainable_adapt)

        self.beta = tf.Variable(beta, dtype=dtype, name="Beta", trainable=False)
        # self.decay_b = np.expand_dims(np.exp(-dt / tau_adaptation),axis=0)
        # self.decay_b = tf.Variable(np.expand_dims(np.exp(-dt / tau_adaptation),axis=0),\
        #                            dtype=dtype, name="rho", trainable=trainable_adapt)
        self.tau_a_max = np.max(tau_adaptation) # constant!
        
        # leaky past dependencies
        self.kappa = tf.Variable(tf.zeros_like(self._decay), trainable=False, name='leaky_past') # dim (n_rec, )
        self.W_back = tf.Variable(tf.zeros_like(self.W_rec), trainable=False, name='W_back')
        self.Wpow = tf.Variable(tf.zeros_like(self.W_rec), trainable=False, name='Wpow')
        self.Wab = tf.zeros_like(self.W_rec[:,:,0]) # will get overwritten in the main code before training starts
        
        self.MDGLpp_mode = MDGLpp_mode

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec, self.n_rec, self.n_rec, self.n_rec]

    @property
    def state_size(self):
        return ALIFStateTuple(v=self.n_rec,
                              z=self.n_rec,
                              b=self.n_rec,
                              i_future_buffer=(self.n_rec, self.n_delay),
                              z_buffer=(self.n_rec, self.n_refractory),
                              vghost=self.n_rec)

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        vghost0 = v0

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return ALIFStateTuple(
            v=v0,
            z=z0,
            b=b0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0,
            vghost=vghost0
        )
    
    def einsum_bi_ijk_to_bjk_(self, x, W):
        h_hat = einsum_bi_ijk_to_bjk(x, W)
        return h_hat
        

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        with tf.name_scope('ALIFcall'):          
            if (self.custom_mode>0): 
                z_stop = tf.stop_gradient(state.z)
            else:
                z_stop = state.z                
                
            if self.custom_mode == 4:
                i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + \
                custom_rec(state.z, self.W_rec, self.W_back)
            elif self.custom_mode <= -1:
                # This is to apply activation derivative approximation only to the s->z branch
                # right before the weighted sum in gradient calculation
                # Exact activation derivative calculation is still used for eligibility trace computation 
                v_scaled = state.v - (self.thr + state.b*self.beta*self.V0)
                dampening = -0.1*tf.cast(self.custom_mode, tf.float32)
                if self.MDGLpp_mode==1:
                    z_stop = rateFunction_1(v_scaled, dampening)
                elif (self.MDGLpp_mode==4) or (self.MDGLpp_mode==5):
                    z_stop = rateFunction_5(v_scaled, dampening, self.psiT)
                elif (self.MDGLpp_mode==2) or (self.MDGLpp_mode==3):
                    z_stop = actFunction(v_scaled, dampening)
                
                if inputs.shape[-1] > self.n_in:   # if truncation
                    inputs1 = inputs[:,:self.n_in]
                    inputs2 = tf.expand_dims(inputs[:,self.n_in:], axis=-1)
                    i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs1, self.W_in) + \
                        inputs2*self.einsum_bi_ijk_to_bjk_(z_stop, self.W_rec) + (1-inputs2)*self.einsum_bi_ijk_to_bjk_(tf.stop_gradient(state.z), self.W_rec)
                else:     # no truncation
                    i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + \
                        self.einsum_bi_ijk_to_bjk_(z_stop, self.W_rec)
            else:
                if inputs.shape[-1] > self.n_in:   # if truncation
                    inputs1 = inputs[:,:self.n_in]
                    inputs2 = tf.expand_dims(inputs[:,self.n_in:], axis=-1)
                    i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs1, self.W_in) + \
                        inputs2*einsum_bi_ijk_to_bjk(z_stop, self.W_rec) + (1-inputs2)*einsum_bi_ijk_to_bjk(tf.stop_gradient(state.z), self.W_rec)
                else:     # no truncation
                    i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + \
                        einsum_bi_ijk_to_bjk(z_stop, self.W_rec)
            self.decay_b = tf.exp(-1/tf.clip_by_value(self.tau_adaptation, 1, 3*self.tau_a_max))
            new_b = self.decay_b * state.b + (1. - self.decay_b) * state.z

            thr = self.thr + new_b * self.beta * self.V0
            
            new_v, new_z, psi, new_vghost = self.LIF_dynamic(
                v=state.v,
                z=state.z,
                z_buffer=state.z_buffer,
                i_future_buffer=i_future_buffer, vghost=state.vghost,
                decay=self._decay,
                thr=thr)

            new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
            new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

            new_state = ALIFStateTuple(v=new_v,
                                       z=new_z,
                                       b=new_b,
                                       i_future_buffer=new_i_future_buffer,
                                       z_buffer=new_z_buffer, 
                                       vghost=new_vghost)
        return [new_z, new_v, thr, psi, new_vghost], new_state