# Source code for the paper Learning the greatest common divisor: explaining transformer predictions

This directory contains the source code for the paper [Learning the greatest common divisor: explaining transformer predictions](https://arxiv.org/abs/2308.15594) (ICLR 2024).

## Environment
* Requirements: Numpy, pyTorch, python 3.8+.
* OS: Tested on Linux Ubuntu, on Windows, add `--window true` to the command line.
* On a SLURM cluster, `--is_slurm_job true`. Multi-gpu training, which allows you to  increase your batch size by sharing it over several GPU requires a SLURM cluster (I doubt you will need it).
* A NVIDIA/CUDA GPU is recommended of you intend to train models: if you do not have one, set `--cpu true` (and be very patient). CPU-only inference works fine.

## Running the programs
To run the program: `python train.py --dump_path MYPATH --exp_name EXPNAME --exp_id EXPID  --parameters (see below)`. 

Training logs will be found in `MYPATH/EXPNAME/EXPID/train.log`, trained models will be `MYPATH/EXPNAME/EXPID/*.pth` . Please make MYPATH an absolute path: relative paths seem not to work on some systems. `--dump_path`and `--exp_name` are mandatory. If `--exp_id`is missing, the program will generate a random one. If `MYPATH/EXPNAME/EXPID`already exists, the program will reload the last saved model, and take it from there (i.e. relaunch an experiment).

To run inference/tests on a trained model : copy the runtime parameters from the corresponding `train.log` file, change the `exp_name` and `exp_id`, and set `--eval_only true` and `--reload_model MODELPATH` (the full path to the saved model).

All models are trained from generated train and test data. Random data generation (and model initialization) are governed by the parameter `--env_base_seed`. A positive seed make the random number generator reproducible (use 42 twice, with the same parameters and you should have the same train and test data, and the same model initialization). A negative seed makes it different on every run.

## Important Parameters

### Data generation

`--base` base used to represent all integers

`--max_int` maximum value of operands (M in the paper), all operands in the training and test set are between 1 and max_int

`--benford` false for uniform sampling of operands (sections 3 and 4), true for log-uniform (section 5)

`--train_uniform_gcd` when true, GCD are distributed uniformly in the training set (section 6)
 
`--train_inverse_dist`  when true, GCD are distributed log-uniformly in the training set (last paragraphs of sections 4 and 5)

`--train_sqrt_dist` `--train_32_dist` other distribution of GCD (1/sqrt(n) and 1/n sqrt(n)) (appendix D.1)

`--mixture` if > 0.0, GCD are dsitributed as a mixture of their natural (1/k**2) distribution, and a uniform distribution (last paragraph of section 3)


### Training and test loops

`--max_len` maximum length of input or output sequence

`--max_output_len` maximum length of output sequence

    

### float16 / AMP API
`--fp16` use float16

`--amp` use amp for variable precision, -1: don't use, >=1 use in fp16, 0 use in fp32

### Transformer architecture (model size, depth and heads)
`--n_enc_layers` layers in the encoder

`--n_dec_layers` layers in the decoder

`--enc_emb_dim` dimensions in the encoder (the FFN hidden layer has 4 times this numbers)

`--dec_emb_dim` dimensions in the decoder (the FFN hidden layer has 4 times this numbers)
 
`--n_enc_heads` attention heads in the encoder (must divide `enc_emb_dim`)

`--n_dec_heads` attention heads in the decoder (must divide `dec_emb_dim`)

`--lstm` `--gru` replace the transformer by an LSTM or GRU (appendix D.5)



## A walk through the code base

`train.py` : the main program, argument parsing and main()

`src/slurm.py` `src/logger.py` `src/utils.py` : various utilities.

`src/trainer.py`: the training loop. Training uses teacher forcing.

`src/evaluator.py`: the test loop, run at the end of every epoch. Generation is auto-regressive.

`src/dataset.py`: the data loader.

`src/optim.py`: code for the various optimisers (on top of those defined by pyTorch, see get_optimizer() for a list). Redefines Adam, with warmup and two scheduling plans (InvSqrt and Cosine).

 `src/model/transformer.py`: the transformer code, initialized in `src/model/__init__.py`

 `src/envs/arithmetic.py`: problem-specific code, and arguments. Example generation is in gen_expr(), test-time evaluation of a transformer prediction in check_predictions(). 

 `src/envs/generators.py`: generation and evaluation routines. generate() is called by gen_expr() (generates a problem instance for this task), evaluate() by check_predictions() (verifies a model predition for this task).

 `src/envs/encoders.py`: integer encoding and decoding. 


## License - Community

GCD is licensed, as per the license found in the LICENSE file.
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## References - citations

Learning the greatest common divisor: explaining transformer predictions

`@misc{charton2023GCD,
  url = {https://arxiv.org/abs/2308.15594},
  author = {Charton, François},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Learning the greatest common divisor: explaining transformer predictions},
  publisher = {arXiv},
  year = {2023}
}`
