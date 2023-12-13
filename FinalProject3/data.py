import gc
import os
import sys
from time import time, sleep
import json
import math
import pickle
from collections import defaultdict

import numpy as np
import numba
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from kaggle_environments import (
    evaluate, make, utils,
    get_episode_replay, list_episodes, list_episodes_for_submission
)

os.listdir("../input")
df_episode_agents = pd.read_csv("../input/meta-kaggle/EpisodeAgents.csv")
recent_submissions = [19352359, 19352272, 19352181, 19352079, 19351928,
                      19335271, 19334971, 19334470, 19334131, 19334038,
                      19325005, 19324861, 19324344, 19324095, 19323940,
                      19311665, 19311618, 19311565, 19311503, 19311396,
                      19287465, 19287233, 19286948, 19286747, 19286415,
                      19278412, 19278217, 19278083, 19277961, 19277822,
                      19252525, 19252439, 19252356, 19252202, 19252130,
                      19237820, 19237695, 19237490, 19237372, 19237108,
                      19224197, 19224008, 19223754, 19223504, 19223418,
                      19208120, 19207921, 19207843, 19207741, 19207633,]
set_recent_submissions = set(recent_submissions)
df_episode_agents = df_episode_agents[df_episode_agents["SubmissionId"].isin(set_recent_submissions)]
df_episode_agents.reset_index(drop=True, inplace=True)
episodes = sorted(df_episode_agents["EpisodeId"].unique().tolist())
set_recent_submission_episode_id_and_indexes = set((df_episode_agents["EpisodeId"] * 2 + df_episode_agents["Index"]).tolist())
del df_episode_agents
gc.collect()
print(f"len(episodes)={len(episodes)}")

data = []
seen_episodes = set()

for episode_id in recent_submissions:
    replay = get_episode_replay(episode_id)
    sleep(1)
    if not replay["wasSuccessful"]:
        continue
    d = json.loads(replay["result"]["replay"])
    if d["statuses"] != ['DONE', 'DONE']:
        continue

    final_rewards = d["rewards"]
    thresholds = d["steps"][0][0]["observation"]["thresholds"]
    actions = []
    last_reward_0 = 0
    last_reward_1 = 0
    rewards = []
    for step in d["steps"][1:]:
        actions.append([step[0]["action"], step[1]["action"]])
        reward_0 = step[0]["reward"]
        reward_1 = step[1]["reward"]
        rewards.append([
            reward_0 - last_reward_0,
            reward_1 - last_reward_1,
        ])
        last_reward_0 = reward_0
        last_reward_1 = reward_1
    dat = {
        "episode_id": episode_id,
        "final_rewards": final_rewards,
        "rewards": rewards,
        "actions": actions,
        "thresholds": thresholds,
    }
    data.append(dat)