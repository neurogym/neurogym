import numpy as np

import neurogym as ngym
from neurogym.core import TrialWrapper


class SideBias(TrialWrapper):
    """Changes the probability of ground truth with block-wise biases.

    Args:
        env: The task environment to wrap (must expose `choices`).
        probs: Explicit probability matrix with shape (n_blocks, n_choices).
               Each row defines the choice probabilities for one block and must sum to 1.0.
               The number of columns (n_choices) must match the number of choices defined in the task
               (i.e., `len(env.choices)`), so that each probability maps to a valid choice.
        block_dur: Duration of each block, with behavior depending on the type:
            - int (≥ 1): Use a fixed number of trials per block
                (e.g., block_dur=20 means each block has exactly 20 trials).
            - float (0 < value < 1): Specify a per-trial probability of switching
                to a new block (e.g., block_dur=0.1 means there's a 10% chance of
                switching blocks after each trial).
            - tuple (low, high): Draw the number of trials per block randomly from
                a uniform distribution over the integer range [low, high] (inclusive).

    Examples:
        - probs=[[0.8, 0.2], [0.2, 0.8], [0.4, 0.6]], block_dur=200
          Stay 200 trials per block, randomly switch to new block after;
        - probs=[[0.8, 0.2], [0.2, 0.8], [0.4, 0.6]], block_dur=0.1
          10% probability per trial to switch to new random block;
        - probs=[[0.8, 0.2], [0.2, 0.8], [0.4, 0.6]], block_dur=(200, 400)
          Random trials in [200, 400] range per block, then switch.
    """

    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": "Changes the probability of ground truth with block-wise biases.",
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(self, env: ngym.TrialEnv, probs: list[list[float]], block_dur: float | tuple[int, int] = 200) -> None:
        super().__init__(env)

        # Validate environment
        if not isinstance(self.task, ngym.TrialEnv):
            msg = "SideBias requires the wrapped task to be a TrialEnv."
            raise TypeError(msg)

        try:
            self.choices = self.task.choices  # type: ignore[attr-defined]
        except AttributeError as e:
            msg = "SideBias requires task to have attribute choices."
            raise AttributeError(msg) from e

        # Reject non-matrix types (no automatic matrix generation)
        if (
            not isinstance(probs, list)
            or not all(isinstance(row, list) for row in probs)
            or not all(isinstance(prob, (float, int)) for row in probs for prob in row)
        ):
            msg = (
                "probs must be a 2D list of lists (matrix) with shape (n_blocks, n_choices),"
                "e.g., probs = [[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]] for n_blocks = 3 and n_choices = 2."
            )
            raise TypeError(msg)

        # Convert to numpy array and validate
        self.choice_prob = np.array(probs, dtype=float)

        # Validate matrix dimensions
        if self.choice_prob.ndim != 2:
            msg = f"probs must be a 2D matrix, got {self.choice_prob.ndim}D array."
            raise ValueError(msg)

        n_blocks, n_choices = self.choice_prob.shape

        if n_choices != len(self.choices):
            msg = (
                f"The number of choices inferred from the probability matrix ({n_choices}) "
                f"does not match the number of choices defined in the task ({len(self.choices)}). "
                "These must be equal to ensure each probability corresponds to a valid choice."
            )
            raise ValueError(msg)

        # Validate that each row sums to 1 (with tolerance for floating point)
        row_sums = np.sum(self.choice_prob, axis=1)
        tolerance = 1e-10
        if not np.allclose(row_sums, 1.0, atol=tolerance):
            invalid_rows = np.where(~np.isclose(row_sums, 1.0, atol=tolerance))[0]
            msg = (
                f"Each row in probs must sum to 1.0. "
                f"Rows {invalid_rows.tolist()} have sums {row_sums[invalid_rows].tolist()}"
            )
            raise ValueError(msg)

        # Validate probabilities are non-negative
        if np.any(self.choice_prob < 0):
            msg = "All probabilities in probs must be non-negative."
            raise ValueError(msg)

        self.n_block = n_blocks
        self.curr_block = self.task.rng.choice(self.n_block)

        # Validate and set block_dur
        self._validate_and_set_block_dur(block_dur)

        # Initialize block state
        self._remaining_trials, self._p_switch = self._new_block_duration()

    def _validate_and_set_block_dur(self, block_dur: float | tuple):
        """Validate block_dur parameter and store specification."""
        self.block_dur = block_dur

        # Case 1: Fixed integer duration
        if isinstance(block_dur, int):
            if block_dur < 1:
                msg = f"if `block_dur` is given as an `int`, the value must be >=1; received {block_dur}."
                raise ValueError(msg)
            return

        # Case 2: Per-trial switch probability
        if isinstance(block_dur, float):
            if not (0 < block_dur < 1):
                msg = f"if `block_dur` is given as a `float`, the value must be between 0 and 1 (exclusive); \
                received {block_dur}"
                raise ValueError(msg)
            return

        # Case 3: Uniform random range
        if isinstance(block_dur, tuple):
            if len(block_dur) != 2:
                msg = (
                    "When specifying block_dur as a tuple, it must contain exactly two elements: (low, high),"
                    "representing the inclusive range of trials per block."
                )
                raise ValueError(msg)

            low, high = block_dur
            if not isinstance(low, int) or not isinstance(high, int):
                msg = "block_dur range values must be integers."
                raise TypeError(msg)

            if low < 1:
                msg = "block_dur range low value must be >= 1."
                raise ValueError(msg)

            if high < low:
                msg = "block_dur range high value must be >= low value."
                raise ValueError(msg)

            return

        # Invalid type
        msg = (
            "block_dur must be one of: "
            "int >= 1 (fixed duration), "
            "0 < float < 1 (switch probability), "
            "or (low, high) tuple of ints (random range)."
        )
        raise TypeError(msg)

    def _new_block_duration(self):
        """Generate duration parameters for the next block.

        Returns:
            tuple: (remaining_trials, p_switch)
                - remaining_trials: int or None (for probability mode)
                - p_switch: float or None (for counter mode)
        """
        block_dur = self.block_dur

        # Fixed integer duration
        if isinstance(block_dur, int):
            return block_dur, None

        # Per-trial switch probability
        if isinstance(block_dur, float):
            return None, block_dur

        # Uniform random range
        if isinstance(block_dur, tuple):
            low, high = block_dur
            duration = self.task.rng.randint(low, high + 1)  # inclusive high
            return int(duration), None

        # This check is in place as a reminder that if the allowed input types change, they are also handled properly.
        msg = f"Unexpected block_dur type: {type(block_dur)}"
        raise TypeError(msg)

    def new_trial(self, **kwargs):
        """Generate new trial with block-based probability biases."""
        # Determine if we should switch blocks
        if self._p_switch is not None:
            # Probability-based switching
            switch_block = self.task.rng.random() < self._p_switch
        else:
            # Counter-based switching
            self._remaining_trials -= 1
            switch_block = self._remaining_trials <= 0

        if switch_block:
            # Switch to a different random block (never same as current)
            if self.n_block > 1:
                # Multiple blocks available - choose different one
                available_blocks = [b for b in range(self.n_block) if b != self.curr_block]
                self.curr_block = self.task.rng.choice(available_blocks)
            # If only 1 block, stay in same block (edge case)

            # Generate new duration/probability for the new block
            self._remaining_trials, self._p_switch = self._new_block_duration()

        # Set trial parameters based on current block
        probs = self.choice_prob[self.curr_block]
        kwargs["ground_truth"] = self.task.rng.choice(self.choices, p=probs)
        kwargs["probs"] = probs

        return self.env.new_trial(**kwargs)
