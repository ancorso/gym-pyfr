import numpy as np
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps, last_len = -np.inf, 0, 0

def get_callback(log_dir):
    def callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward, last_len
        # Print stats every 1000 calls
        if (n_steps + 1) % 100 == 0:
          # Evaluate policy performance
          x, y = ts2xy(load_results(log_dir), 'timesteps')
          if len(x) > last_len:
              last_len = len(x)
              mean_reward = np.mean(y[-100:])
              print(x[-1], 'timesteps')
              print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > best_mean_reward:
                  best_mean_reward = mean_reward
                  # Example for saving best model
                  print("Saving new best model")
                  _locals['self'].save(log_dir + '/best_model.pkl')
        n_steps += 1
        return True
    return callback
