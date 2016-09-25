
# Image caption with semantic attention 
### note that this repository are mainly borrowed from [neuraltalk2](https://github.com/karpathy/neuraltalk2), hats off to Karpathy, what a great job he has done! And the model implemented here is from [image caption with semantic attention](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/You_Image_Captioning_With_CVPR_2016_paper.pdf), Quanzeng You et al. CVPR2016.
### without regularization on attention weights

current results table

|beam_size | Bleu_1        |Bleu_2    |Bleu_3 |Bleu_4 |METEOR |  CIDEr|
| ------------- |:-------------:| :-------------:|:-------------:|:-------------:|:-------------:|-----:|
| 2      | 0.884 | 0.726 |0.58 |0.46 |0.308 |1.214 |
| 3      | 0.891 | 0.739 | 0.597 |0.479 |0.311 |1.239 |
| 4      | 0.891 | 0.742 |0.601 |0.484 |0.312 |1.244 |
| 5      | 0.892 | 0.743 |0.603 |0.488 |0.313 |1.249 |
| 7      | 0.893 | 0.744 |0.605 |0.489 |0.313 |1.25 |

### with regularization on attention weights
current result: to be updated..
L1 loss on output attention weights(seems not improve too much):

|beam_size | Bleu_1        |Bleu_2    |Bleu_3 |Bleu_4 |METEOR |  CIDEr|
| ------------- |:-------------:| :-------------:|:-------------:|:-------------:|:-------------:|-----:|
| 7      | 0.898 | 0.751 |0.614 |0.498 |0.315 |1.26 |

Attention Weights Criterion on attention weights(to be updated):

only regularization at output attention model:

|beam_size | Bleu_1        |Bleu_2    |Bleu_3 |Bleu_4 |METEOR |  CIDEr|
| ------------- |:-------------:| :-------------:|:-------------:|:-------------:|:-------------:|-----:|
| 7      | 0.90 | 0.754 |0.615 |0.498 |0.314 |1.258 |
* (may add comment later, below is the comment from neuraltalk2, shoule remove it in the near future)



### Requirements


#### For evaluation only

This code is written in Lua and requires [Torch](http://torch.ch/). If you're on Ubuntu, installing Torch in your home directory may look something like: 

```bash
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
```

See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using [LuaRocks](https://luarocks.org/) (which already came with the Torch install). In particular:

```bash
$ luarocks install nn
$ luarocks install nngraph 
$ luarocks install image 
```

We're also going to need the [cjson](http://www.kyne.com.au/~mark/software/lua-cjson-manual.html) library so that we can load/save json files. Follow their [download link](http://www.kyne.com.au/~mark/software/lua-cjson.php) and then look under their section 2.4 for easy luarocks install.

If you'd like to run on an NVIDIA GPU using CUDA (which you really, really want to if you're training a model, since we're using a VGGNet), you'll of course need a GPU, and you will have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Then get the `cutorch` and `cunn` packages:

```bash
$ luarocks install cutorch
$ luarocks install cunn
```

If you'd like to use the cudnn backend (the pretrained checkpoint does), you also have to install [cudnn](https://github.com/soumith/cudnn.torch). First follow the link to [NVIDIA website](https://developer.nvidia.com/cuDNN), register with them and download the cudnn library. Then make sure you adjust your `LD_LIBRARY_PATH` to point to the `lib64` folder that contains the library (e.g. `libcudnn.so.7.0.64`). Then git clone the `cudnn.torch` repo, `cd` inside and do `luarocks make cudnn-scm-1.rockspec` to build the Torch bindings.

#### For training

If you'd like to train your models you will need [loadcaffe](https://github.com/szagoruyko/loadcaffe), since we are using the VGGNet. First, make sure you follow their instructions to install `protobuf` and everything else (e.g. `sudo apt-get install libprotobuf-dev protobuf-compiler`), and then install via luarocks:

```bash
luarocks install loadcaffe
```

Finally, you will also need to install [torch-hdf5](https://github.com/deepmind/torch-hdf5), and [h5py](http://www.h5py.org/), since we will be using hdf5 files to store the preprocessed data.

Phew! Quite a few dependencies, sorry no easy way around it :\

### I'd like to distribute my GPU trained checkpoints for CPU

Use the script `convert_checkpoint_gpu_to_cpu.lua` to convert your GPU checkpoints to be usable on CPU. See inline documentation for why this separate script is needed. For example:

```bash
th convert_checkpoint_gpu_to_cpu.lua gpu_checkpoint.t7
```

write the file `gpu_checkpoint.t7_cpu.t7`, which you can now run with `-gpuid -1` in the eval script.

### License

BSD License.

### Acknowledgements

Parts of this code were written in collaboration with my labmate [Justin Johnson](http://cs.stanford.edu/people/jcjohns/). 

I'm very grateful for [NVIDIA](https://developer.nvidia.com/deep-learning)'s support in providing GPUs that made this work possible.

I'm also very grateful to the maintainers of Torch for maintaining a wonderful deep learning library.
