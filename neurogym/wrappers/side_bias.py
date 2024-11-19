import numpy as np

import neurogym as ngym


class SideBias(ngym.TrialWrapper):
    """Changes the probability of ground truth.

    Args:
        prob: Specifies probabilities for each choice. Within each block,the
            probability should sum up to 1. (def: None, numpy array (n_block,
            n_choices))
        block_dur: Number of trials per block. (def: 200, int)
    """

    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": "Changes the probability of ground truth.",
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(self, env, probs=None, block_dur=200) -> None:
        super().__init__(env)
        try:
            self.choices = self.task.choices
        except AttributeError as e:
            msg = "SideBias requires task to have attribute choices."
            raise AttributeError(msg) from e
        if not isinstance(self.task, ngym.TrialEnv):
            msg = "Task has to be TrialEnv."
            raise TypeError(msg)
        if probs is None:
            msg = "Please provide choices probabilities."
            raise ValueError(msg)
        if isinstance(probs, float | int):
            mat = np.eye(len(self.choices)) * probs
            mat[mat == 0] = 1 - probs
            self.choice_prob = mat
        else:
            self.choice_prob = np.array(probs)
        if self.choice_prob.shape[1] != len(self.choices):
            msg = (
                f"The number of choices {self.choice_prob.shape[1]} inferred from prob mismatches "
                f"{len(self.choices)} inferred from choices."
            )
            raise ValueError(msg)

        self.n_block = self.choice_prob.shape[0]
        self.curr_block = self.task.rng.choice(range(self.n_block))
        self.block_dur = block_dur

    def new_trial(self, **kwargs):
        # change rep. prob. every self.block_dur trials
        if self.task.num_tr % self.block_dur == 0:
            curr_block = self.curr_block
            while curr_block == self.curr_block:
                curr_block = self.task.rng.choice(range(self.n_block))
            self.curr_block = curr_block
        probs = self.choice_prob[self.curr_block]
        kwargs = {}
        kwargs["ground_truth"] = self.task.rng.choice(self.choices, p=probs)
        return self.env.new_trial(**kwargs)
