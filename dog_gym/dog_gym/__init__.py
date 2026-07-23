from gymnasium.envs.registration import register

register(
    id='Dog-Stand-v0',
    entry_point='dog_gym.envs.dog_env:DogEnv',
    kwargs={'task': 'stand'},
)

register(
    id='Dog-Walk-v0',
    entry_point='dog_gym.envs.dog_env:DogEnv',
    kwargs={'task': 'walk'},
)
