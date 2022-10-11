'''
Code adapted for training a RNN using ModProp to perform a delayed XOR task. 

The overall ModProp framework proposed is "communicating the credit information 
via cell-type-specific neuromodulators and processing it at the receiving cells 
via pre-determined temporal filtering taps." 

Remarks:        
    - If you also train with BPTT and three-factor (by changing FLAGS.custom_mode) across many runs, 
    it should reproduce the performance ordering between rules as reported in the paper 
    - Performance for each rule may fluctuate across runs, so one should repeat across many runs, 
    as the focus is on performance trend across many runs 
    - Typically the worst run (or few runs) is ignored for every rule in plotting the learning curves 
    - As mentioned, this is a proof-of-concept study and future work involves testing ModProp across
    a wide range of architecture; as such, there would be no performance guarantee if some network 
    parameters were changed (e.g. sparsity and thr)

Current approximations are proof of concept for the framework, and better approximations 
can be developed as a part of the future work 

Code built on top of https://github.com/IGITUGraz/LSNN-official
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

'''

from lsnn.toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lsnn.toolbox.file_saver_dumper_no_h5py import save_file, load_file, get_storage_path_reference
from plotting_tools import *

from neuron_models import tf_cell_to_savable_dict, exp_convolve, ALIF, LIF, Gru, MDGL_output
from rewiring_tools_NP2 import weight_sampler, rewiring_optimizer_wrapper, rewiring_optimizer_wrapper_NP
from lsnn.toolbox.tensorflow_utils import tf_downsample
import json

FLAGS = tf.app.flags.FLAGS

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
# tf.app.flags.DEFINE_string('resume', '', 'path to the checkpoint of the form "results/script_name/2018_.../session"')
tf.app.flags.DEFINE_string('model', 'LSNN', 'Network model, do not change')
tf.app.flags.DEFINE_bool('save_data', False, 'whether to save simulation data in result folder')
tf.app.flags.DEFINE_bool('downsampled', False, 'Do not change')
## 
tf.app.flags.DEFINE_integer('batch_size', 32, 'size of the minibatch')
tf.app.flags.DEFINE_integer('delay_window', 700, 'Delay Window')
tf.app.flags.DEFINE_integer('n_in', 2, 'number of input units, do not change') 
tf.app.flags.DEFINE_float('n_multiplier', 1, 'multiplier for number of recurrent neurons')
tf.app.flags.DEFINE_integer('n_regular', 120, 'number of regular units in the recurrent layer')
tf.app.flags.DEFINE_integer('n_adaptive', 0, 'number of adaptive units in the recurrent layer, do not change')
tf.app.flags.DEFINE_integer('n_iter', 3000, 'number of training iterations')
tf.app.flags.DEFINE_integer('n_delay', 1, 'maximum synaptic delay, do not change')
tf.app.flags.DEFINE_integer('n_ref', 5, 'number of refractory steps, not used')
tf.app.flags.DEFINE_integer('lr_decay_every', 300, 'decay learning rate every lr_decay_every steps')
tf.app.flags.DEFINE_integer('print_every', 20, 'frequency of printing training progress')
tf.app.flags.DEFINE_float('custom_mode', -2.5, '0 if BPTT, 1 if three-factor; -x, for ModProp with mu=-0.1x')
tf.app.flags.DEFINE_integer('trainable_adapt_mode', 0, 'Do not change')
tf.app.flags.DEFINE_integer('MDGL_buflen', -1, 'Buffer length S for MDGL++; -1 for full length')
tf.app.flags.DEFINE_integer('MDGLpp_mode', 3, 'ModProp variants: 2) cell-specific feedback W, 3) type-specific W')
tf.app.flags.DEFINE_integer('ntype_EI', 1, 'Number of E types and I types, do not change')
##
tf.app.flags.DEFINE_float('beta', 1.8, 'Scaling constant of the adaptive threshold, not used')
tf.app.flags.DEFINE_float('tau_a', 700, 'Adaptation time constant mean, not used')
tf.app.flags.DEFINE_float('tau_a_range', 0, 'Adaptation time constant range, not used')
tf.app.flags.DEFINE_float('tau_v', 100, 'Membrane time constant of units, do not change')
tf.app.flags.DEFINE_float('thr', 0., 'Baseline threshold voltage, do not change')
tf.app.flags.DEFINE_float('learning_rate', 0.0005, 'Base learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'Decaying factor')
tf.app.flags.DEFINE_float('reg_l2', 5e-4, 'regularization coefficient l2')
tf.app.flags.DEFINE_float('rewiring_temperature', 0., 'regularization coefficient')
tf.app.flags.DEFINE_float('proportion_excitatory', 0.8, 'proportion of excitatory neurons, do not change')
tf.app.flags.DEFINE_float('in_high', 2.0, 'amplitude for 1')
##
tf.app.flags.DEFINE_bool('verbose', True, 'Print many info during training')
tf.app.flags.DEFINE_bool('neuron_sign', True,
                         "If rewiring is active, this will fix the sign of neurons (Dale's law)")
tf.app.flags.DEFINE_bool('crs_thr', False, 'Do not change')
tf.app.flags.DEFINE_float('rewiring_connectivity', 0.99, 'max connectivity limit in the network, do not change')
tf.app.flags.DEFINE_float('wout_connectivity', 0.99, 'similar to above but for output weights, do not change')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization used in rewiring (irrelevant without rewiring)')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Parameter necessary to approximate the spike derivative, not used')
# Analog values are fed to only single neuron

# The following arguments should be fixed
FLAGS.crs_thr=False    
FLAGS.downsampled = False
FLAGS.trainable_adapt_mode = 0
FLAGS.n_in = 2
FLAGS.n_adaptive = 0
FLAGS.n_delay = 1
FLAGS.ntype_EI = 1
FLAGS.tau_v = 100
FLAGS.model = 'LSNN'
FLAGS.rewiring_connectivity = 0.99
FLAGS.wout_connectivity = 0.99
FLAGS.thr = 0
FLAGS.proportion_excitatory = 0.8

assert (FLAGS.MDGLpp_mode==2) or (FLAGS.MDGLpp_mode==3), 'FLAGS.MDGLpp_mode must be 2 or 3'
assert (FLAGS.custom_mode <=2), 'FLAGS.custom_mode must be at most 2'

FLAGS.n_regular = int(FLAGS.n_regular*FLAGS.n_multiplier)
FLAGS.n_adaptive = int(FLAGS.n_adaptive*FLAGS.n_multiplier)

n_unit = FLAGS.n_regular + FLAGS.n_adaptive

dt = 1.  # Time step is by default 1 ms
fix_window = 0
cue_window = 100
delay_window = FLAGS.delay_window
T = tr_len = fix_window+2*cue_window+delay_window

batch_size = FLAGS.batch_size  # the batch size 
n_output_symbols = 2 # two classes for now

if FLAGS.trainable_adapt_mode < 0:
    FLAGS.beta = 0.0
    
if FLAGS.trainable_adapt_mode == -2:
    FLAGS.tau_v = 0.01

# Define the flag object as dictionnary for saving purposes
_, storage_path, flag_dict = get_storage_path_reference(__file__, FLAGS, './results/', flags=False,
                                                        comment=len(FLAGS.comment) > 0)
if FLAGS.save_data:
    os.makedirs(storage_path, exist_ok=True)
    save_file(flag_dict, storage_path, 'flag', 'json')
    print('saving data to: ' + storage_path)
print(json.dumps(flag_dict, indent=4))

# Sign of the neurons
if 0 < FLAGS.rewiring_connectivity and FLAGS.neuron_sign:
    n_excitatory_in = int(FLAGS.proportion_excitatory * FLAGS.n_in) + 1
    n_inhibitory_in = FLAGS.n_in - n_excitatory_in
    in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
    np.random.shuffle(in_neuron_sign)

    n_excitatory = int(FLAGS.proportion_excitatory * (n_unit)) + 1
    n_inhibitory = n_unit - n_excitatory
    rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
else:
    if not (FLAGS.neuron_sign == False): print(
        'WARNING: Neuron sign is set to None without rewiring but sign is requested')
    in_neuron_sign = None
    rec_neuron_sign = None

# Define the network
tau_v = FLAGS.tau_v  

w_adj_dict = None
wout_adj = None
    
if FLAGS.model == 'LSNN':
    # We set beta == 0 to some of the neurons. Those neurons then behave like LIF neurons (without adaptation).
    # And this is how we achieve a mixture of LIF and ALIF neurons in the LSNN model.
    beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
    tau_a_array=np.random.uniform(low=FLAGS.tau_a-FLAGS.tau_a_range, high=FLAGS.tau_a+FLAGS.tau_a_range, size=(n_unit,))
    #tau_a_array = np.random.normal(FLAGS.tau_a, FLAGS.tau_a_var, size=(FLAGS.n_regular+FLAGS.n_adaptive,))
    cell = ALIF(n_in=FLAGS.n_in, n_rec=n_unit, tau=tau_v, n_delay=FLAGS.n_delay,
                n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=tau_a_array, beta=beta, thr=FLAGS.thr,
                rewiring_connectivity=FLAGS.rewiring_connectivity,
                in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                dampening_factor=FLAGS.dampening_factor, custom_mode=FLAGS.custom_mode, trainable_adapt_mode=FLAGS.trainable_adapt_mode,
                w_adj=w_adj_dict, MDGLpp_mode=FLAGS.MDGLpp_mode, task='dlyXOR'
                )
elif FLAGS.model == 'Gru': # Not used
    cell = Gru(n_in=FLAGS.n_in, n_rec=n_unit,
                rewiring_connectivity=FLAGS.rewiring_connectivity,
                in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign, custom_mode=FLAGS.custom_mode,
                w_adj=w_adj_dict
                )
else:
    raise NotImplementedError("Unknown model: " + FLAGS.model)
cell.psiT = tf.Variable(np.zeros((FLAGS.batch_size, n_unit)), trainable=False, name='psiT', dtype=tf.float32)
# cell.psiT = tf.Variable(np.zeros((1, n_unit)), trainable=False, name='psiT', dtype=tf.float32)

    
    
def get_data_dict():
    """
    Generate the dictionary to be fed when running a tensorflow op. 
    
    i1 and i2: two streams of inputs, 40 Hz if +, 0 otherwise
    i4: 10 Hz throughout the trial
    """
    
    # Initialize target and input cue matrices 
    target_num = -1*np.ones((FLAGS.batch_size,))
    
    # cue received by i1 and i2
    cue_batch = np.random.randint(0,2,size=(FLAGS.batch_size,2))  
   
    input_stack=np.zeros((FLAGS.batch_size,tr_len,FLAGS.n_in))
    
    # Get spike encoding and target for each trial  
    def get_input_stack(cue):  # spikes per example
        # i4_spike = np.random.poisson(0.01, (tr_len,10))         
                       
        in_high = FLAGS.in_high
        in_low = 0.02 #in_high/2 #0.02 
        gauss_std = cue_window/2/2        
        time_steps = np.linspace(-int(cue_window/2),int(cue_window/2),cue_window)
        gauss_in = in_high*np.expand_dims(np.exp(-(time_steps**2)/2/gauss_std**2), axis=-1) + in_low
        i1_spike = in_low*np.ones((tr_len,1))  
        i2_spike = in_low*np.ones((tr_len,1)) 
        
        # cue b4 delay                                  
        tstamps = np.array(range(fix_window, fix_window+cue_window))        
        if cue[0]==1:
            i1_spike[tstamps,:] = gauss_in #in_high
        
        # cue after delay
        tstamps = np.array(range(tr_len-cue_window, tr_len))
        if cue[1]==1:
            i2_spike[tstamps,:] = gauss_in #in_high
            
        input_stack = np.concatenate((i1_spike,i2_spike),1)
        target_dir=int(cue[0]==cue[1])   # 1:match, 0:mismatch
        
        return input_stack, target_dir

    # loop through trials across batches
    for tr in range(len(cue_batch)):
        cue_num = cue_batch[tr,:]             
        input_stack[tr,:,:], target_num[tr] = get_input_stack(cue_num)
        
    # add some noise
    input_stack += np.random.normal(0.0, 0.01, size=input_stack.shape)   
    
    if (FLAGS.MDGL_buflen > 0) and (FLAGS.custom_mode < 0):
        midT = tr_len - FLAGS.MDGL_buflen
        inputs2 = np.zeros((FLAGS.batch_size, tr_len, 1))
        inputs2[:,midT:] = 1 # select richer grad propagation after midT
        input_stack = np.concatenate((input_stack, inputs2), axis=2)
    
    # transform target one hot from batch x classes to batch x time x classes
    data_dict = {inputs: input_stack, targets: target_num}
    return data_dict, cue_batch
    

  
# Generate input
if (FLAGS.custom_mode > 1) and (FLAGS.model == 'Gru'):
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in*2),
                                  name='InputSpikes')  # MAIN input spike placeholder
elif (FLAGS.MDGL_buflen > 0) and (FLAGS.custom_mode<0):
    inputs = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, None, FLAGS.n_in+1),
                                  name='InputSpikes') 
else:
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                                  name='InputSpikes')  # MAIN input spike placeholder

targets = tf.placeholder(dtype=tf.int64, shape=(None,),
                         name='Targets')  # Lists of target characters of the recall task

# create outputs and states
outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
final_state = states[-1]

if FLAGS.model == 'LSNN':
    z, v, b, psi, _ = outputs
else:
    z, psi = outputs
z_regular = z[:, :, :FLAGS.n_regular]
z_adaptive = z[:, :, FLAGS.n_regular:]


with tf.name_scope('ClassificationLoss'):
    #psp_decay = np.exp(-dt / FLAGS.tau_v)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
    psp = z #exp_convolve(z, decay=psp_decay)
    n_neurons = z.get_shape()[2]

    # Define the readout weights
    if (0 < FLAGS.wout_connectivity):
        w_out, w_out_sign, w_out_var, _ = weight_sampler(n_unit, n_output_symbols,
                                                          FLAGS.wout_connectivity,
                                                          neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols], dtype=tf.float32)
    b_out = tf.get_variable(name='out_bias', shape=[n_output_symbols], initializer=tf.zeros_initializer(), dtype=tf.float32)

    # Define Wab    
    Wrec = cell.W_rec[:,:,0]
    if (FLAGS.MDGLpp_mode == 3):
        n_per_type_I = int(n_inhibitory/FLAGS.ntype_EI) 
        n_per_type_E = int(n_excitatory/FLAGS.ntype_EI)
        if n_inhibitory % FLAGS.ntype_EI:
            inh_idx = list(range(0,n_inhibitory,n_per_type_I)[:-1])
        else:
            inh_idx = list(range(0,n_inhibitory,n_per_type_I))
        if n_excitatory % FLAGS.ntype_EI:             
            exc_idx = list(range(n_inhibitory,n_unit,n_per_type_E)[:-1])
        else:
            exc_idx = list(range(n_inhibitory,n_unit,n_per_type_E))
        exc_idx.append(n_unit)
        tp_idx_ = np.concatenate((np.array(inh_idx), np.array(exc_idx)))
        tp_idx = np.stack((tp_idx_[:-1], tp_idx_[1:]),axis=1)
        n_type = len(tp_idx)
        for ii in range(n_type):
            for jj in range(n_type):
                W_block = Wrec[tp_idx[ii][0]:tp_idx[ii][1],tp_idx[jj][0]:tp_idx[jj][1]]                           
                Wav = tf.reduce_mean(W_block)                 
                if jj==0: # new row
                    th_row = Wav * tf.ones_like(W_block) 
                else:
                    th_row = tf.concat([th_row, Wav*tf.ones_like(W_block)], axis=1)
                if jj==(n_type-1): # finished a row 
                    if ii==0:
                        th_ = th_row
                    else: 
                        th_ = tf.concat([th_, th_row], axis=0) 
        cell.Wab = th_ 
        
    # Define the loss function   
    if (FLAGS.custom_mode>=2) and (FLAGS.model != 'Gru'): # customized MDGL grad
        if FLAGS.custom_mode == 2:
            Wab = tf.transpose(cell.W_rec[:,:,0]) #jp
        elif (FLAGS.custom_mode == 3):
            Wab = tf.transpose(Wab)
                
        out = MDGL_output(z, w_out, Wab, psi, cell._decay) + b_out
    else:
        out = einsum_bij_jk_to_bik(z, w_out) + b_out
        
    Y_predict = out[:, -1, :]  
    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=Y_predict))

    # Define the accuracy
    Y_predict_num = tf.argmax(Y_predict, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, Y_predict_num), dtype=tf.float32))
 
# Target regularization
with tf.name_scope('RegularizationLoss'):
    # # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    # # regularization_f0 = FLAGS.reg_rate / 1000
    # # loss_regularization = tf.reduce_sum(tf.square(av - regularization_f0)) * FLAGS.reg
    reg_l2 = FLAGS.reg_l2*(tf.nn.l2_loss(cell.w_in_val)+tf.nn.l2_loss(cell.w_rec_val)+tf.nn.l2_loss(w_out))
    loss_regularization = reg_l2
    

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay)  # Op to decay learning rate

    loss = loss_recall +loss_regularization 

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)    

    if 0 < FLAGS.rewiring_connectivity:
        mProp_tuple = (FLAGS.MDGLpp_mode==3 and FLAGS.custom_mode<0, cell)
        train_step = rewiring_optimizer_wrapper(optimizer, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
                                            FLAGS.rewiring_connectivity,
                                            global_step=global_step,
                                            var_list=tf.trainable_variables(), mProp_tuple=mProp_tuple)
    else:
        train_step = optimizer.minimize(loss=loss, global_step=global_step)

# Real-time plotting
# saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# if FLAGS.resume:
#     saver.restore(sess, FLAGS.resume)
#     print("Model restored.")

# Store some results across iterations
test_loss_list = []
# test_loss_with_reg_list = []
test_error_list = []
#tau_delay_list = []
#taub_list = []
training_time_list = []
time_to_ref_list = []

# Dictionaries of tensorflow ops to be evaluated simultaneously by a session
results_tensors = {'loss': loss,
                   'loss_recall': loss_recall,
                   'accuracy': accuracy,
                   'av': av,
                   'learning_rate': learning_rate,

                   'w_in_val': cell.W_in,
                   'w_rec_val': cell.w_rec_val,
                   'w_out': w_out,
                   }
if FLAGS.model == 'LSNN':
    results_tensors['b_out'] = b_out

plot_result_tensors = {'inputs': inputs,
                       'z': z,
                       'psp': psp,
                       'Y_predict': Y_predict,
                       'z_regular': z_regular,
                       'z_adaptive': z_adaptive,
                       'targets': targets}
if FLAGS.model == 'LSNN':
    plot_result_tensors['b_con'] = b

t_train = 0
for k_iter in range(FLAGS.n_iter):

    # Decaying learning rate
    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # Print some values to monitor convergence
    if np.mod(k_iter, FLAGS.print_every) == 0:

        val_dict, input_stack = get_data_dict()
        results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)

        # Storage of the results
        # test_loss_with_reg_list.append(results_values['loss_reg'])
        test_loss_list.append(results_values['loss_recall'])
        test_error_list.append(results_values['accuracy'])
        # if FLAGS.model == 'LSNN':
        #     taub_list.append(sess.run(cell.tau_adaptation))
        # else:
        #     taub_list.append(-99)
        training_time_list.append(t_train)

        print(
            '''Iteration {}, validation accuracy {:.3g} '''
                .format(k_iter, test_error_list[-1],))


        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max


        firing_rate_stats = get_stats(results_values['av'] * 1000) # no *1000 for rate

        # some connectivity statistics
        rewired_ref_list = ['w_in_val', 'w_rec_val', 'w_out']
        non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
        sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
        empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)
        empirical_connectivities = [nz / size for nz, size in zip(non_zeros, sizes)]

        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t average {:.0f} +- std {:.0f} (over neurons)
            connectivity (total {:.3g})\t W_in {:.3g} \t W_rec {:.2g} \t\t w_out {:.2g}
            number of non zero weights \t W_in {}/{} \t W_rec {}/{} \t w_out {}/{}

            classification loss {:.2g} 
            learning rate {:.2g} \t training op. time {:.2g}s
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                empirical_connectivity,
                empirical_connectivities[0], empirical_connectivities[1], empirical_connectivities[2],
                non_zeros[0], sizes[0],
                non_zeros[1], sizes[1],
                non_zeros[2], sizes[2],
                results_values['loss_recall'],
                results_values['learning_rate'], t_train,
            ))

        # Save files result
        if FLAGS.save_data:
            results = {
                'error': test_error_list[-1],
                'loss': test_loss_list[-1],
                'error_list': test_error_list,
                'loss_list': test_loss_list,
                'time_to_ref': time_to_ref_list,
                'training_time': training_time_list,
                #'tau_delay_list': tau_delay_list,
                #'taub': taub_list[-1],
                'flags': flag_dict,
            }
            
            save_file(results, storage_path, 'results', file_type='json')

    # train
    t0 = time()
    train_dict, input_stack = get_data_dict()
    
    final_state_value, _ = sess.run([final_state, train_step], feed_dict=train_dict)
    t_train = time() - t0 
    

# if FLAGS.interactive_plot:
#     update_mnist_plot(ax_list, fig, plt, cell, FLAGS, plot_results_values)

# del sess