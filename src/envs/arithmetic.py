# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import math
import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators


from torch.utils.data import DataLoader
from src.dataset import EnvDataset

from ..utils import bool_flag


SPECIAL_WORDS = ["<eos>", "<pad>", "<sep>", "(", ")"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]

logger = getLogger()


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class ArithmeticEnvironment(object):

    TRAINING_TASKS = {"arithmetic"}

    def __init__(self, params):
        self.max_len = params.max_len
        
        self.base = params.base
        self.max_uniform = params.max_uniform

        dims = []
        max_dim =  2
        tensor_dim =  1
        self.output_encoder = encoders.PositionalInts(params.base)

        self.input_encoder = encoders.NumberArray(params, max_dim, 'V', tensor_dim)
        self.generator = generators.Sequence(params, dims)

        # vocabulary
        self.words = SPECIAL_WORDS + sorted(list(
            set(self.input_encoder.symbols+self.output_encoder.symbols)
        ))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        self.sep_index = params.sep_index = 2
        
        logger.info(f"words: {self.word2id}")

    def input_to_infix(self, lst):
        return ''.join(lst)
        
    def output_to_infix(self, lst):
        return ''.join(lst)
        
    def gen_expr(self, data_type=None, task=None):
        """
        Generate pairs of problems and solutions.
        Encode this as a prefix sentence
        """
        gen = self.generator.generate(self.rng, data_type)
        if gen is None:
            return None
        x_data, y_data = gen
        # encode input
        x = self.input_encoder.encode(x_data)
        # encode output
        y = self.output_encoder.encode(y_data)
        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None
        return x, y

    def decode_class(self, i):
        if i>=1000:
            return str(i//1000)+"-"+str(i%1000)
        return str(i)

    def code_class(self, xi, yi):
        top = 0
        v = self.output_encoder.decode(yi)
        if v is None:
            return 0
        if v > self.max_uniform and v > 100:
            v=101
        return 1000*top + v

    def check_prediction(self, src, tgt, hyp):
        w = self.output_encoder.decode(hyp)
        if w is None:
            return 0,0,0,0
        if len(hyp) == 0 or len(tgt) == 0:
            return 0, 0, 0, w
        #t = self.output_encoder.decode(tgt)
        #if t is None:
        #    return 0,0,0,w
        #if w == t:
        #    return 1, 1, 1, w
        if hyp == tgt:
            return 1, 1, 1, w
        return 0, 0, 0, w

    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
            type = "train"
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"]
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][1 if data_type == "valid" else 2]
            ),
            size=size,
            type=data_type,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--maxint", type=int, default=1000000, help="Maximum value of integers"
        )
        parser.add_argument(
            "--benford", type=bool_flag, default=False, help="Logarithmic distribution of integers"
        )
        parser.add_argument(
            "--train_uniform_gcd", type=bool_flag, default=False, help="Uniformly distributed gcd in train set, 1 to max uniform"
        )
        parser.add_argument(
            "--train_inverse_dist", type=bool_flag, default=False, help="gcd distributed as 1/K (instead of inverse squares) in train set, 1 to max_uniform"
        )
        parser.add_argument(
            "--train_sqrt_dist", type=bool_flag, default=False, help="gcd distributed as 1/sqrt(K) (instead of inverse squares) in train set, 1 to max_uniform"
        )
        parser.add_argument(
            "--train_32_dist", type=bool_flag, default=False, help="gcd distributed as 1/K^3/2 (instead of inverse squares) in train set, 1 to max_uniform"
        )
        parser.add_argument(
            "--max_inverse", type=int, default=100, help="Maximum value of inverse distributed gcd"
        )
        
        parser.add_argument(
            "--max_uniform", type=int, default=100, help="Maximum value of uniformly distributed gcd"
        )
        parser.add_argument(
            "--test_uniform_gcd", type=bool_flag, default=True, help="Uniformly distributed gcd in test set, 1 to max uniform"
        )

        parser.add_argument(
            "--mixture", type=float, default=-1.0, help="Proportion of uniformly sampled outcomes"
        )       
       
        parser.add_argument(
            "--base", type=int, default=1000, help="Encoding base"
        )
        
