import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from neurogym.wrappers import TrialHistoryV2, monitor
import neurogym as ngym
from neurogym.envs.raposo.network import CustomRNNPolicy

# --- Constants ---
n_networks = 100
save_path = '/Users/lexotto/Documents_Mac/Stage/UVA/Code/neurogym/neurogym/envs/raposo/data'
target_accuracy = 80
test_trials = 1024
threshold = 0.2

# Create directories
os.makedirs(f"{save_path}/good_runs", exist_ok=True)
os.makedirs(f"{save_path}/bad_runs", exist_ok=True)

# --- Custom evaluation function replicating your logic ---
def evaluate_model(model, env, task, tau=100, dt=2, minibatch_size=4):
    test_data = task.generate_trials(np.random.default_rng(), dt, test_trials)
    test_inputs = torch.Tensor(test_data['inputs'])  # shape: [N, T, input_dim]

    test_outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, test_trials, minibatch_size)):
            batch = test_inputs[i:i+minibatch_size]
            output, _ = model.policy.features_extractor.rnn(batch, tau=tau, dt=dt)
            test_outputs.append(output.numpy())

    test_output = np.concatenate(test_outputs, axis=0)

    out_diff = test_output[:, test_data['phases']['stimulus'], 1] - test_output[:, test_data['phases']['stimulus'], 0]
    decision_time = np.argmax(np.abs(out_diff) > threshold, axis=1)

    out_diff_onset = test_output[:, test_data['phases']['stimulus'][0], 1] - test_output[:, test_data['phases']['stimulus'][0], 0]
    valid_start = np.nonzero(np.abs(out_diff_onset) <= threshold)[0]
    choice_made = np.nonzero(np.sum(np.abs(out_diff) > threshold, axis=1) != 0)[0]

    valid_and_made = np.intersect1d(valid_start, choice_made)

    choice = (out_diff[valid_and_made, decision_time[valid_and_made]] > 0).astype(np.int_)
    true_choice = test_data['choice'][valid_and_made]
    correct = np.sum(choice == true_choice)

    accuracy = 100 * correct / len(valid_and_made) if len(valid_and_made) > 0 else 0

    # Prepare stats
    stats = {
        'n_analyzed_trials': len(valid_and_made),
        'n_choices_made': len(choice),
        'n_test_correct': correct,
        'accuracy': accuracy
    }

    return accuracy, stats, test_output, test_data

# --- Training loop ---
for n in range(1, n_networks + 1):
    print(f"\n🧠 Training model {n}/{n_networks}")

    # Create env
    env = ngym.make("CustomRaposoTask-v0")
    env = TrialHistoryV2(env, history_size=1)
    env = monitor.Monitor(env, folder=f"{save_path}/monitor_{n}", force=True)
    vec_env = DummyVecEnv([lambda: env])

    # Get reference to the task object
    task = env.envs[0].unwrapped  # Extract the raw task for test trial generation

    # Create and train model
    model = PPO(
        policy=CustomRNNPolicy,
        env=vec_env,
        verbose=0,
    )
    model.learn(total_timesteps=100_000)

    # Evaluate model
    acc, stats, test_output, test_trials_dict = evaluate_model(model, vec_env, task)

    # Save to correct folder
    result_dir = f"{save_path}/{'good_runs' if acc >= target_accuracy else 'bad_runs'}/{n}"
    os.makedirs(result_dir, exist_ok=True)

    model_name = "model" if acc >= target_accuracy else "model_not_converged"
    model.save(f"{result_dir}/{model_name}")

    # Save accuracy to .txt
    with open(f"{result_dir}/accuracy.txt", "w") as f:
        f.write(f"Final Accuracy: {acc:.2f}%\n")

    # Save training stats
    with open(f"{result_dir}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save test output
    np.save(f"{result_dir}/test_output.npy", test_output)

    # Save test trials
    with open(f"{result_dir}/test_trials.pkl", "wb") as f:
        pickle.dump(test_trials_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved model {n} to {result_dir}")
    vec_env.close()
