import os
import pickle
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from neurogym.wrappers import TrialHistoryV2, monitor
import neurogym as ngym

# Make sure these are imported so your env registers
from neurogym.envs.raposo.raposo_task import RaposoTask
from neurogym.envs.raposo.network import CustomRNNPolicy

n_networks = 100
save_path = '/Users/lexotto/Documents_Mac/Stage/UVA/Code/neurogym/neurogym/envs/raposo/data'
target_accuracy = 80
test_trials = 1024

os.makedirs(f"{save_path}/good_runs", exist_ok=True)
os.makedirs(f"{save_path}/bad_runs", exist_ok=True)

def evaluate_model(model, env, n_trials=1024):
    obs = env.reset()
    correct = 0
    total = 0
    test_output = []
    test_trials_list = []
    for _ in range(n_trials):
        action, _ = model.predict(obs, deterministic=True)
        test_output.append(action)
        test_trials_list.append(obs)
        obs, reward, done, info = env.step(action)
        correct += int(reward[0] > 0)
        total += 1
        if done:
            obs = env.reset()
    accuracy = 100 * correct / total
    return accuracy, np.array(test_output), test_trials_list

for n in range(1, n_networks + 1):
    print(f"\n🧠 Training model {n}/{n_networks}")

    # Define environment for SB3
    env = ngym.make("CustomRaposoTask-v0")
    env = TrialHistoryV2(env)
    env = monitor.Monitor(env, folder=f"{save_path}/monitor_{n}")
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class=CustomRNNExtractor,
        features_extractor_kwargs=dict(features_dim=64))

    # Create and train PPO agent
    model = PPO(
        policy=CustomRNNPolicy,
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1
        )

    model.learn(total_timesteps=100_000)

    # Evaluate and save
    acc, test_output, test_trials = evaluate_model(model, vec_env, n_trials=test_trials)

    result_dir = f"{save_path}/{'good_runs' if acc >= target_accuracy else 'bad_runs'}/{n}"
    os.makedirs(result_dir, exist_ok=True)
    model_name = "model" if acc >= target_accuracy else "model_not_converged"
    model.save(f"{result_dir}/{model_name}")

    # Save accuracy
    with open(f"{result_dir}/accuracy.txt", "w") as f:
        f.write(f"Final Accuracy: {acc:.2f}%\n")

    # Save test output
    np.save(f"{result_dir}/test_output.npy", test_output)
    # Save test trials (as pickle)
    with open(f"{result_dir}/test_trials.pkl", "wb") as f:
        pickle.dump(test_trials, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save minimal training stats
    stats = {
        'accuracy': acc,
        'n_trials_tested': test_trials
    }
    with open(f"{result_dir}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved model {n} to {result_dir}")
    vec_env.close()
