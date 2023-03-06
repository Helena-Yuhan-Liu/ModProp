# ModProp

This repository (in TensorFlow 1.12) implements ModProp, a biologically plausible learning rule for recurrent neural networks (RNNs) inspired by ubiquitous cell-type-specific modulatory signals [1]. The code is developed on top of the LSNN repository (https://github.com/IGITUGraz/LSNN-official) [2]. 

The overall ModProp framework proposed is "communicating the credit information via cell-type-specific neuromodulators and processing it at the receiving cells via pre-determined temporal filtering taps." Specifically, ModProp proposes a biologically plausible approximation to the exact gradient computed by real-time recurrent learning (RTRL). ModProp involves the following two approximations:

1. Approximating feedback weights using cell-type-specific modulatory weights (instead of cell-specific) to mimick the cell-type-specificity of neuromodulation [1]. 

2. Approximates the activation derivative (for the nonlocal gradient terms only) so that the credit signal can be processed via pre-determined temporal filter taps.

This code demonstrates a proof of concept for cell-type-specific neuromodulation for communicating credit signal (inspired from neuropeptide signaling molecules). Because it is just a proof of concept, it has several shortcomings to be improved in the future:

1. The current ModProp formulation depends heavily on fine-tuning the hyperparameter µ; without properly tuning µ, training can go numerically unstable. Future work involves modifying ModProp with an adaptive µ for better stability and accuracy. 
2. The current ModProp formulation does not work well when activity is not stationary; future work involves extending ModProp to nonstationary data. 
3. Future work also involves testing ModProp across a broad range of tasks and architectures (e.g. sparse connections, neuronal threshold adaptation, spiking neurons), with the hope of improving ModProp during that process. 

Additional remarks: 
1. Please be advised that performance for the same rule can fluctuate across different runs (with different random weight initializations). It is possible for ModProp to perform similarly as MDGL/e-prop on some runs. However, the focus is on the trend across many runs, so one should repeat each rule with at least several runs. 
2. It is important not to include the activation derivative approximation in the eligibility trace term, i.e. the approximation only applies right before they pass through the recurrent weights. 
3. For using cell-type-specific weight matrices, this code uses automatic differentiation in TensorFlow (to avoid numerical instabilities) by replacing recurrent weights with Wab right before the gradient is computed, and then setting the recurrent weights to their original value before applying the Adam optimizer. However, such in-place gradient modification is not supported in PyTorch. 

## Usage

The main code is in the `bin/` folder. You can use the following command to run:
``sh run_dlyXOR.sh``

The command above runs ``delayedXOR_task.py`` 8 times, which contains the code to setup and train a RNN to solve a delayed XOR task. Each iteration should take about 2 hours to complete, so 8 runs should take about less than a day to complete. Inside the same folder, you may also find ``plot_curves.py`` that contains the code to plot the saved results. 

The folder 'lsnn/' contains the source code retained from the lsnn package [1]. 

## Installation

The installation instruction is copied from (https://github.com/IGITUGraz/LSNN-official), and please refer to that repository for troubleshooting steps. The code is compatible with python 3.4 to 3.7 and TensorFlow 1.7 to 1.14 (CPU and GPU versions).

> You can run the training scripts **without installation** by temporarily including the repo directory
> in your python path like so: `` PYTHONPATH=. python3 bin/tutorial_sequential_mnist_with_LSNN.py`` 

From the main folder run:  
`` pip3 install --user .``  
To use GPUs one should also install it:
 ``pip3 install --user tensorflow-gpu``.

## License

Please refer to LICENSE for copyright details


## References

[1] Stephen J. Smith, Uygar Sumbul, Lucas T. Graybuck, Forrest Collman, Sharmishtaa Seshamani, Rohan Gala, Olga Gliko, Leila Elabbady, Jeremy A. Miller, Trygve E. Bakken, Jean Rossier, Zizhen Yao, Ed Lein, Hongkui Zeng, Bosiljka Tasic, and Michael Hawrylycz. Single-cell transcriptomic evidence for dense intracortical neuropeptide networks. eLife, 8, nov 2019.

[2] Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, and Wolfgang Maass. Long short-term memory and learning-to-learn in networks of spiking neurons. In: 32nd Conference on Neural Information Processing Systems. 2018, pp. 787-797.
