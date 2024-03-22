# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np

from .utils import to_cuda


logger = getLogger()


def idx_to_infix(env, idx, input=True):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.id2word[wid] for wid in idx]
    infix = env.input_to_infix(prefix) if input else env.output_to_infix(prefix)
    return infix


def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV

    src = [env.id2word[wid] for wid in eq["src"]]
    tgt = [env.id2word[wid] for wid in eq["tgt"]]
    hyp = [env.id2word[wid] for wid in eq["hyp"]]

    # update hypothesis
    eq["src"] = env.input_to_infix(src)
    eq["tgt"] = tgt
    eq["hyp"] = hyp
    try:
        m, s1, s2, nb = env.check_prediction(src, tgt, hyp)
    except Exception:
        m = 0
        s1 = 0
        s2 = 0
        nb = 0
    eq["is_valid"] = m
    eq["is_valid2"] = s1
    eq["is_valid3"] = s2
    eq["is_valid4"] = nb if not nb is None else 0
    return eq


class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations.

        """
        params = self.params
        scores = OrderedDict({"epoch": self.trainer.epoch})

        # save statistics about generated data
        if params.export_data:
            scores["total"] = self.trainer.total_samples
            return scores
        
        if params.test_uniform_gcd:
            data_type_list = ["valid", "test"]
        else:
            data_type_list = ["valid"]

        with torch.no_grad():
            for data_type in data_type_list:
                for task in params.tasks:
                    if params.beam_eval:
                        self.enc_dec_step_beam(data_type, task, scores)
                    else:
                        self.enc_dec_step(data_type, task, scores)
        return scores

    def enc_dec_step(self, data_type, task, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        max_beam_length = params.max_output_len + 2
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        encoder.eval()
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        decoder.eval()
        assert params.eval_verbose in [0, 1,2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in env.TRAINING_TASKS

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(
                params.dump_path, f"eval.{data_type}.{task}.{scores['epoch']}"
            )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):  # FC A revoir
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res["hyps"]])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\n"
                s += f"src={res['src']}\ntgt={res['tgt']}\n"
                for hyp, score, valid in res["hyps"]:
                    if score is None:
                        s += f"{int(valid)} {hyp}\n"
                    else:
                        s += f"{int(valid)} {score :.3e} {hyp}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # stats
        xe_loss = 0
        n_valid = torch.zeros(10000, dtype=torch.long)
        n_total = torch.zeros(10000, dtype=torch.long)
        n_perfect_match = 0
        n_correct = 0
        n_valid_d1 = 0
        n_valid_d2 = 0
        n_valid_d3 = 0
        n_pairs = torch.zeros((102,102), dtype=torch.long)

        # iterator
        iterator = self.env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
        )
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # cuda
            x1_, len1_, x2, len2 = to_cuda(x1, len1, x2, len2)
            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            bs = len(len1_)

            # forward / loss
            if params.lstm:
                _, hidden = encoder("fwd", x=x1_, lengths=len1_, causal=False)
                decoded, _ = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=hidden,
                )
                word_scores, loss = decoder(
                    "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
                )
            else:
                encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
                decoded = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=encoded.transpose(0, 1),
                    src_len=len1_,
                )
                word_scores, loss = decoder(
                    "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
                )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()
            n_perfect_match += valid.sum().item()

            # export evaluation details
            beam_log = {}
            for i in range(len(len1)):
                out_offset = 1
                src = idx_to_infix(env, x1[0 : len1[i] - 1, i].tolist(), True)
                tgt = idx_to_infix(env, x2[out_offset : len2[i] - 1, i].tolist(), False)
                if valid[i]:
                    beam_log[i] = {"src": src, "tgt": tgt, "hyps": [(tgt, None, True)]}
                    result = (nb_ops[i] % 1000) if (nb_ops[i] % 1000) < 101 else 101
                    n_pairs[result][result] += 1

            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                display_logs(beam_log, offset=n_total.sum().item() - bs)
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total.sum().item()}/{eval_size}) Found "
                f"{bs - len(invalid_idx)}/{bs} valid top-1 predictions. "
                f"Generating solutions ..."
            )

            # generate
            if params.lstm:
                generated, _ = decoder.generate(
                    hidden,
                    len1_,
                    max_len=max_beam_length,
                )
            else:
                generated, _ = decoder.generate(
                    encoded.transpose(0, 1),
                    len1_,
                    max_len=max_beam_length,
                )
                generated=generated.transpose(0, 1)
            
            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            inputs = []
            for i in range(len(generated)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                out_offset = 1
                inputs.append(
                    {
                        "i": i,
                        "src": x1[0 : len1[i] - 1, i].tolist(),
                        "tgt": x2[out_offset : len2[i] - 1, i].tolist(),
                        "hyp": generated[i][out_offset:].tolist(),
                        "task": task,
                    }
                )

            # check hypotheses with multiprocessing
            outputs = []
            if params.windows is True:
                for inp in inputs:
                    outputs.append(check_hypothesis(inp))
            else:
                with ProcessPoolExecutor(max_workers=20) as executor:
                    for output in executor.map(check_hypothesis, inputs, chunksize=1):
                        outputs.append(output)
            
            # read results
            for i in range(bs):
                # select hypotheses associated to current equation
                gens = [o for o in outputs if o["i"] == i]            
                assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2)
                assert (i in beam_log) == valid[i]
                if len(gens) == 0:
                    continue

                assert len(gens) == 1
                # source / target
                gen = gens[0]
                src = gen["src"]
                tgt = gen["tgt"]
                beam_log[i] = {"src": src, "tgt": tgt, "hyps": []}

                # sanity check
                assert (
                    gen["src"] == src
                    and gen["tgt"] == tgt
                    and gen["i"] == i
                )

                # if hypothesis is correct, and we did not find a correct one before
                is_valid = gen["is_valid"]
                is_valid2 = gen["is_valid2"]
                is_valid3 = gen["is_valid3"]
                is_valid4 = gen["is_valid4"]
                is_b_valid = is_valid > 0
                if not valid[i]:
                    result = (nb_ops[i] % 1000) if (nb_ops[i] % 1000) < 101 else 101
                    prediction = 101 if (is_valid4 is None or is_valid4 > 101) else is_valid4
                    n_pairs[result][prediction] += 1
                if is_valid > 0 and not valid[i]:
                    n_correct += 1
                    n_valid[nb_ops[i]] += 1
                    valid[i] = 1
                
                if not valid[i]:
                    if is_valid2 > 0:
                        n_valid_d1 += 1
                    if is_valid3 > 0:
                        n_valid_d2 += 1
                    if is_valid4 > 0:
                        n_valid_d3 += 1

                # update beam log
                beam_log[i]["hyps"].append((gen["hyp"], None, is_b_valid))  # gen["score"], is_b_valid))

            # valid solutions found with beam search
            logger.info(
                f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses."
            )

            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                display_logs(beam_log, offset=n_total.sum().item() - bs)

        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
            f"equations were evaluated correctly."
        )

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_acc"] = 100.0 * _n_valid / _n_total
        scores[f"{data_type}_{task}_perfect"] = 100.0 * n_perfect_match / _n_total
        scores[f"{data_type}_{task}_correct"] = (
            100.0 * (n_perfect_match + n_correct) / _n_total
        )
        scores[f"{data_type}_{task}_acc_d1"] = (
            100.0 * (n_perfect_match + n_valid_d1) / _n_total
        )
        scores[f"{data_type}_{task}_acc_d2"] = (
            100.0 * (n_perfect_match + n_valid_d2) / _n_total
        )
        scores[f"{data_type}_{task}_acc_d3"] = (
            100.0 * (n_perfect_match + n_valid_d3) / _n_total
        )
                        
        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            e = env.decode_class(i)
            scores[f"{data_type}_{task}_acc_{e}"] = (
                100.0 * n_valid[i].item() / max(n_total[i].item(), 1)
            )
            if n_valid[i].item() > 0:
                logger.info(
                    f"{e}: {n_valid[i].item()} / {n_total[i].item()} "
                    f"({100. * n_valid[i].item() / max(n_total[i].item(), 1):.2f}%)"
                )
        if data_type == "test":
            logger.info(f"{data_type} predicted pairs")
            for i in range(102):
                for j in range(102):
                    if n_pairs[i][j].item() >= 10:
                        logger.info(f"{i}-{j}: {n_pairs[i][j].item()} ({100. * n_pairs[i][j].item() / n_pairs[i].sum().item():2f}%)")

    def enc_dec_step_beam(self, data_type, task, scores, size=None):
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        params = self.params
        env = self.env
        max_beam_length = params.max_output_len + 2
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in env.TRAINING_TASKS

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(
                params.dump_path, f"eval.beam.{data_type}.{task}.{scores['epoch']}"
            )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res["hyps"]])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\n"
                s += f"src={res['src']}\ntgt={res['tgt']}\n"
                for hyp, score, valid in res["hyps"]:
                    if score is None:
                        s += f"{int(valid)} {hyp}\n"
                    else:
                        s += f"{int(valid)} {score :.3e} {hyp}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # stats
        xe_loss = 0
        n_valid = torch.zeros(10000, dtype=torch.long)
        n_total = torch.zeros(10000, dtype=torch.long)
        n_perfect_match = 0
        n_correct = 0
        n_valid_d1 = 0
        n_valid_d2 = 0
        n_valid_d3 = 0

        # iterator
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
        )
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            bs = len(len1)

            # forward
            if params.lstm:
                encoded, hidden = encoder("fwd", x=x1_, lengths=len1_, causal=False)
                decoded, _ = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=hidden,
                )
            else:
                encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
                decoded = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=encoded.transpose(0, 1),
                    src_len=len1_,
                )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
            )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()
            n_perfect_match += valid.sum().item()

            # save evaluation details
            beam_log = {}
            for i in range(len(len1)):
                src = idx_to_infix(env, x1[1 : len1[i] - 1, i].tolist(), True)
                tgt = idx_to_infix(env, x2[1 : len2[i] - 1, i].tolist(), False)
                if valid[i]:
                    beam_log[i] = {"src": src, "tgt": tgt, "hyps": [(tgt, None, True)]}

            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                display_logs(beam_log, offset=n_total.sum().item() - bs)
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total.sum().item()}/{eval_size}) Found "
                f"{bs - len(invalid_idx)}/{bs} valid top-1 predictions. "
                f"Generating solutions ..."
            )

            # generate
            if params.lstm:
                _, _, generations = decoder.generate_beam(
                    hidden,
                    len1_,
                    beam_size=params.beam_size,
                    length_penalty=params.beam_length_penalty,
                    early_stopping=params.beam_early_stopping,
                    max_len=max_beam_length,
                )
            else:
                _, _, generations = decoder.generate_beam(
                    encoded.transpose(0, 1),
                    len1_,
                    beam_size=params.beam_size,
                    length_penalty=params.beam_length_penalty,
                    early_stopping=params.beam_early_stopping,
                    max_len=max_beam_length,
                )

            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            inputs = []
            for i in range(len(generations)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                for j, (score, hyp) in enumerate(
                    sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)
                ):
                    inputs.append(
                        {
                            "i": i,
                            "j": j,
                            "score": score,
                            "src": x1[1 : len1[i] - 1, i].tolist(),
                            "tgt": x2[1 : len2[i] - 1, i].tolist(),
                            "hyp": hyp[1:].tolist(),
                            "task": task,
                        }
                    )

            # check hypotheses with multiprocessing
            outputs = []
            if params.windows is True:
                for inp in inputs:
                    outputs.append(check_hypothesis(inp))
            else:
                with ProcessPoolExecutor(max_workers=20) as executor:
                    for output in executor.map(check_hypothesis, inputs, chunksize=1):
                        outputs.append(output)

            # read results
            for i in range(bs):

                # select hypotheses associated to current equation
                gens = sorted([o for o in outputs if o["i"] == i], key=lambda x: x["j"])
                assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2) and (
                    i in beam_log
                ) == valid[i]
                if len(gens) == 0:
                    continue

                # source / target
                src = gens[0]["src"]
                tgt = gens[0]["tgt"]
                beam_log[i] = {"src": src, "tgt": tgt, "hyps": []}

                curr_correct = 0
                curr_d1 = 0
                curr_d2 = 0
                curr_d3 = 0
                curr_valid = 0
        
                # for each hypothesis
                for j, gen in enumerate(gens):

                    # sanity check
                    assert (
                        gen["src"] == src
                        and gen["tgt"] == tgt
                        and gen["i"] == i
                        and gen["j"] == j
                    )

                    # if hypothesis is correct, and we did not find a correct one before
                    is_valid = gen["is_valid"]
                    is_valid2 = gen["is_valid2"]
                    is_valid3 = gen["is_valid3"]
                    is_valid4 = gen["is_valid4"]
                    is_b_valid = is_valid > 0
                    if is_valid > 0 and not valid[i]:
                        curr_correct = 1
                        curr_valid = 1

                    if not valid[i]:
                        if is_valid2 > 0:
                            curr_d1 = 1
                        if is_valid3 > 0:
                            curr_d2 = 1
                        if is_valid4 > 0:
                            curr_d3 = 1

                    # update beam log
                    beam_log[i]["hyps"].append((gen["hyp"], gen["score"], is_b_valid))

                if not valid[i]:
                    n_correct += curr_correct
                    n_valid_d1 += curr_d1
                    n_valid_d2 += curr_d2
                    n_valid_d3 += curr_d3
                    valid[i] = curr_valid
                    n_valid[nb_ops[i]] += curr_valid

            # valid solutions found with beam search
            logger.info(
                f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses."
            )

            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                display_logs(beam_log, offset=n_total.sum().item() - bs)

        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
            f"equations were evaluated correctly."
        )

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_beam_acc"] = 100.0 * _n_valid / _n_total
        scores[f"{data_type}_{task}_perfect"] = 100.0 * n_perfect_match / _n_total
        scores[f"{data_type}_{task}_correct"] = (
            100.0 * (n_perfect_match + n_correct) / _n_total
        )
        scores[f"{data_type}_{task}_beam_acc_d1"] = (
            100.0 * (n_perfect_match + n_valid_d1) / _n_total
        )
        scores[f"{data_type}_{task}_beam_acc_d2"] = (
            100.0 * (n_perfect_match + n_valid_d2) / _n_total
        )
        scores[f"{data_type}_{task}_beam_acc_d3"] = (
            100.0 * (n_perfect_match + n_valid_d3) / _n_total
        )
        
        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            e = env.decode_class(i)
            logger.info(
                f"{e}: {n_valid[i].item()} / {n_total[i].item()} "
                f"({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)"
            )
            scores[f"{data_type}_{task}_beam_acc_{e}"] = (
                100.0 * n_valid[i].item() / max(n_total[i].item(), 1)
            )
