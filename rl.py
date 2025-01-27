import gym
import numpy as np
from stable_baselines3 import PPO

# Defina um ambiente personalizado para simplificação de reações
class ReactionSimplificationEnv(gym.Env):
    def __init__(self, reactions):
        super(ReactionSimplificationEnv, self).__init__()
        self.reactions = reactions
        self.current_reaction = 0
        self.action_space = gym.spaces.Discrete(2)  # Ações: simplificar ou não
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)  # Exemplo de espaço de observação

    def reset(self):
        self.current_reaction = 0
        return self._get_observation()

    def step(self, action):
        # Lógica para simplificar a reação
        reward = 1 if action == 1 else 0  # Recompensa por simplificar
        done = self.current_reaction >= len(self.reactions) - 1
        self.current_reaction += 1
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Retorna uma representação da reação atual
        return np.random.random(10)  # Exemplo de observação

# Cria o ambiente
reactions = [
    "A + B -> C",
    "C + D -> E",
    "E + F -> G"
]
env = ReactionSimplificationEnv(reactions)

# Treina o agente com PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Testa o agente
obs = env.reset()
for _ in range(len(reactions)):
    action, _ = model.predict(obs)
    obs, _, _, _ = env.step(action)
    print("Ação:", action)