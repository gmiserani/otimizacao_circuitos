import gym
import numpy as np
from stable_baselines3 import PPO

class ReactionSimplificationEnv(gym.Env):
    def __init__(self, reactions):
        super(ReactionSimplificationEnv, self).__init__()
        self.reactions = reactions
        self.current_reaction = 0
        self.simplified_reactions = [] 
        self.action_space = gym.spaces.Discrete(2) 
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(reactions),), dtype=np.float32) 

    def reset(self):
        self.current_reaction = 0
        self.simplified_reactions = []
        return self._get_observation()

    def step(self, action):
        """
        Lógica de execução do ambiente.
        """
        done = self.current_reaction >= len(self.reactions) - 1
        reward = 0

        if action == 1:
            self.simplified_reactions.append(self.reactions[self.current_reaction])
            reward = 1  

        self.current_reaction += 1
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = np.zeros(len(self.reactions))
        if self.current_reaction < len(self.reactions):
            obs[self.current_reaction] = 1
        return obs

    def get_simplified_reactions(self):
        """
        Combina as reações simplificadas em uma única reação consolidada.
        """
        all_reagents = set()
        final_product = None

        for reaction in self.simplified_reactions:
            reagents, product = reaction.split(" -> ")
            all_reagents.update(reagents.split(" + "))
            final_product = product 

        simplified_reaction = " + ".join(sorted(all_reagents)) + " -> " + final_product
        return simplified_reaction

reactions = [
    "A + B -> C",
    "C + D -> E",
    "E + F -> G"
]
env = ReactionSimplificationEnv(reactions)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for _ in range(len(reactions)):
    action, _ = model.predict(obs)
    obs, _, _, _ = env.step(action)
    print("Ação:", "Simplificar" if action == 1 else "Não simplificar")

print("\nReações Simplificadas:", env.simplified_reactions)
print("Reação Consolidada:", env.get_simplified_reactions())
