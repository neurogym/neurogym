from neurogym import _SB3_INSTALLED
from neurogym.envs.native.annubes import AnnubesEnv
from neurogym.wrappers import Monitor


def test_activation_traces(n_trials: int = 10):
    """Tests for neuron activation traces."""
    session: dict[str | tuple[str], float] = {"auditory": 0.5, "visual": 0.5}
    catch_prob = 0.25

    env = AnnubesEnv(session, catch_prob)
    mon = Monitor(env, name=f"NeuroGym Monitor | {env.__class__.__qualname__}")

    if not _SB3_INSTALLED:
        # TODO: Make tests run with generic PyTorch models.
        return

    from sb3_contrib import RecurrentPPO

    # A simple model whose activations we are going to monitor
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=mon,
        verbose=0,
        policy_kwargs={"lstm_hidden_size": 64},
    )

    # Extract the actor net from the policy and
    # register it with the monitor.
    actor = model.policy.get_submodule("lstm_actor")

    am = mon.record_activations(actor, "Actor net")

    # Evaluate the policy, which will populate the activation trace history.
    mon.evaluate_policy(n_trials, model)

    n_traces = len(am.history["hidden"])
    hidden_shape = am.history["hidden"][1][0].shape
    batch_size, neuron_count = hidden_shape[0], hidden_shape[1]
    expected_steps = int(mon.tmax / mon.dt)

    assert n_traces == n_trials, f"Wrong number of recorded activation traces (expected {n_trials}, got {n_traces})"
    for hist_length in am.history["hidden"]:
        assert len(hist_length) <= expected_steps, (
            f"Wrong number of steps in the traces (expected <={expected_steps}, got {hist_length})"
        )
    assert batch_size == 1, f"Wrong batch size (expected 1, got {batch_size})"
    assert neuron_count == actor.hidden_size, (
        f"Wrong number of neurons in the trace (expected {actor.hidden_size}, got {neuron_count})"
    )
