
# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
import numpy as np
import math
from logging import getLogger

logger = getLogger()


class Generator(ABC):
    def __init__(self, params):
        super().__init__()

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass

# empty for now
class Sequence(Generator):
    def __init__(self, params, dims):
        super().__init__(params)

        self.maxint = params.maxint
        self.dims = dims
        self.benford = params.benford
        self.train_uniform_gcd = params.train_uniform_gcd
        self.test_uniform_gcd = params.test_uniform_gcd
        self.max_uniform = params.max_uniform
        self.max_inverse = params.max_inverse

        self.mixture = params.mixture
        self.train_inverse_dist = params.train_inverse_dist
        self.train_sqrt_dist = params.train_sqrt_dist
        self.train_32_dist = params.train_32_dist
        
        self.inverse_dist = np.zeros(self.max_inverse)
        sum = 0
        if self.train_sqrt_dist:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / math.sqrt(i + 1)
                sum += 1 / math.sqrt(i + 1)
        elif self.train_32_dist:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / (i + 1) * math.sqrt(i + 1)
                sum += 1 / (i + 1) * math.sqrt(i + 1)
        else:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / (i + 1)
                sum += 1 / (i + 1)
        self.inverse_dist = self.inverse_dist / sum
        
    def integer_sequence(self, len, rng, type=None, max=None):
        maxint = self.maxint if max is None else max
        if type == "train" and self.benford:
            lgs = math.log10(maxint)*rng.rand(len)
            return np.int64(10 ** lgs)
        return rng.randint(1, maxint + 1, len)


    def generate(self, rng, type=None):
        mix = rng.rand() if (type == "train" and self.mixture > 0.0) else 1.0
        if type == "train" and (self.train_inverse_dist or self.train_sqrt_dist):
            out = rng.choice(range(1, self.max_inverse + 1), p=self.inverse_dist)
            gc = 2
            while gc > 1:
                inp = self.integer_sequence(2, rng, type, self.maxint // out)
                gc = math.gcd(inp[0],inp[1])
            inp[0] = out * inp[0] // gc
            inp[1] = out * inp[1] // gc
        elif (type == "train" and (self.train_uniform_gcd or mix < self.mixture)) or type == "test":
            out = rng.randint(1, self.max_uniform + 1)
            gc = 2
            while gc > 1:
                inp = self.integer_sequence(2, rng, type, self.maxint // out)
                gc = math.gcd(inp[0],inp[1])
            inp[0] = out * inp[0] // gc
            inp[1] = out * inp[1] // gc
        else:
            inp = self.integer_sequence(2, rng, type)
            out = math.gcd(inp[0], inp[1])
        return inp, out

    def evaluate(self, src, tgt, hyp):
        return 0, 0, 0, 0
