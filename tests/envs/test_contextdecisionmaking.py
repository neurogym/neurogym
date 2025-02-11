import pytest
from neurogym.envs.contextdecisionmaking import ContextDecisionMaking
import numpy as np


@pytest.fixture
def default_env():
    """Create a default ContextDecisionMaking environment."""
    env = ContextDecisionMaking()
    return env


@pytest.mark.parametrize("use_expl_context", [True, False])
def test_env_initialization(use_expl_context):
    """Test that environment initializes with expected parameters."""
    env = ContextDecisionMaking(use_expl_context=use_expl_context)
    assert env.use_expl_context == use_expl_context
    assert env.impl_dim_ring == 2
    assert env.sigma > 0
    assert isinstance(env.cohs, list)
    assert len(env.cohs) > 0
    assert env.dt == 100


@pytest.mark.parametrize("use_expl_context", [True, False])
def test_detailed_observation_space(use_expl_context):
    """Test the detailed structure of observation spaces for both context modes.

    This test checks:
    1. The shape of the observation space matches environment configuration
    2. The names and indices assigned to each dimension
    """
    env = ContextDecisionMaking(use_expl_context=use_expl_context)

    if not use_expl_context:
        expected_implicit_dim = (
            1 + 2 * env.impl_dim_ring
        )  # fixation + stimuli for both modalities
        assert env.observation_space.shape == (expected_implicit_dim,)

        expected_obs_space = {"fixation": 0}
        for i in range(env.impl_dim_ring):
            expected_obs_space[f"stimulus{i + 1}_mod1"] = i + 1
            expected_obs_space[f"stimulus{i + 1}_mod2"] = i + 1 + env.impl_dim_ring
        assert env.observation_space.name == expected_obs_space
    else:
        n_basic_inputs = 5  # fixation + 2 stimuli per modality
        n_context_inputs = len(env.contexts)  # number of context inputs
        expected_explicit_dim = n_basic_inputs + n_context_inputs
        assert env.observation_space.shape == (expected_explicit_dim,)

        expected_obs_space = {
            "fixation": 0,
            "stim1_mod1": 1,
            "stim2_mod1": 2,
            "stim1_mod2": 3,
            "stim2_mod2": 4,
            "context1": 5,
            "context2": 6,
        }
        assert env.observation_space.name == expected_obs_space


@pytest.mark.parametrize("use_expl_context", [True, False])
def test_detailed_action_space(use_expl_context):
    """Test the detailed structure of action spaces for both context modes.

    This test checks:
    1. The number of possible actions matches environment configuration
    2. The names and values assigned to each action
    """
    env = ContextDecisionMaking(use_expl_context=use_expl_context)

    if not use_expl_context:
        expected_n_actions = 1 + env.impl_dim_ring  # fixation + choices
        assert env.action_space.n == expected_n_actions

        expected_action_space = {
            "fixation": 0,
            "choice": list(range(1, env.impl_dim_ring + 1)),
        }
        assert env.action_space.name == expected_action_space
    else:
        expected_n_choices = len(env.choices)
        assert env.action_space.n == 1 + expected_n_choices  # fixation + choices

        expected_action_space = {"fixation": 0, "choice1": 1, "choice2": 2}
        assert env.action_space.name == expected_action_space


@pytest.mark.parametrize("use_expl_context", [True, False])
def test_new_trial_generation(use_expl_context):
    """Test new trial generation and structure."""
    env = ContextDecisionMaking(use_expl_context=use_expl_context)
    trial = env.new_trial()

    assert "ground_truth" in trial
    assert "context" in trial
    assert "coh_1" in trial
    assert "coh_2" in trial

    assert trial["ground_truth"] in env.choices
    assert trial["context"] in env.contexts
    assert trial["coh_1"] in env.cohs
    assert trial["coh_2"] in env.cohs


@pytest.mark.parametrize("impl_dim_ring", [2, 4, 8])
def test_implicit_ring_dimensions(impl_dim_ring):
    """Test different ring dimensions for implicit context."""
    env = ContextDecisionMaking(use_expl_context=False, impl_dim_ring=impl_dim_ring)

    assert len(env.theta) == impl_dim_ring
    assert len(env.choices) == impl_dim_ring
    assert env.action_space.n == impl_dim_ring + 1  # choices + fixation


@pytest.mark.parametrize("use_expl_context", [True, False])
def test_step_mechanics(use_expl_context):
    """Test basic stepping mechanics and reward structure."""
    env = ContextDecisionMaking(use_expl_context=use_expl_context)
    env.reset()

    # Test abort during fixation
    assert env.in_period("fixation"), "Environment should start in fixation period"
    _, reward, _, _, info = env.step(1)  # Non-fixation action
    assert reward == env.rewards["abort"], (
        "Should get abort penalty for breaking fixation"
    )

    # Test correct trial sequence
    env.reset()
    trial = env.trial
    correct_choice = trial["ground_truth"]
    assert 1 <= correct_choice <= 2, "Ground truth should be a valid choice index"

    # First maintain fixation
    while env.in_period("fixation"):
        _, reward, _, _, info = env.step(0)
        assert reward == 0, "Should get 0 reward for maintaining fixation"
        if info["new_trial"]:
            break

    # Then maintain fixation through stimulus and delay
    while not env.in_period("decision"):
        _, reward, _, _, info = env.step(0)
        if info["new_trial"]:
            break

    # If we reached decision period, test correct choice
    if env.in_period("decision"):
        _, reward, _, _, info = env.step(correct_choice)
        assert reward == env.rewards["correct"], (
            "Should get correct reward for right choice"
        )
        assert info["new_trial"], "Trial should end after decision"


@pytest.mark.parametrize("use_expl_context", [True, False])
def test_observation_noise(use_expl_context):
    """Test that noise is properly added to stimulus observations."""
    env = ContextDecisionMaking(use_expl_context=use_expl_context)

    def get_stimulus_obs(env):
        """Get an observation during the stimulus period."""
        obs, _ = env.reset()
        # Step through fixation until we reach stimulus
        while env.in_period("fixation"):
            obs, _, _, _, info = env.step(0)
            if info["new_trial"]:
                return None
        return obs  # First observation in stimulus period

    obs1 = get_stimulus_obs(env)
    env.new_trial()
    obs2 = get_stimulus_obs(env)

    assert not np.array_equal(obs1, obs2), (
        "Stimulus period observations should differ due to noise"
    )


@pytest.mark.parametrize(
    "timing",
    [
        {"stimulus": 500},
        {"delay": 800},
        {"decision": 200},
    ],
)
def test_timing_configuration(timing):
    """Test that timing parameters are properly configured."""
    env = ContextDecisionMaking(timing=timing)

    for period, duration in timing.items():
        assert env.timing[period] == duration
