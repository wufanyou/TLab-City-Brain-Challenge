import pickle
import gym
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
import sys

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
import gym_cfg
import pandas as pd

LINE_TO_ACTION = {
    1: [1, 5],
    2: [2, 5],
    3: [0],
    4: [3, 6],
    5: [4, 6],
    6: [0],
    7: [1, 7],
    8: [2, 7],
    9: [0],
    10: [3, 8],
    11: [4, 8],
    12: [0],
}

ACTION_TO_LINE = {
    1: [1, 7],
    2: [2, 8],
    3: [4, 10],
    4: [5, 11],
    5: [1, 2],
    6: [4, 5],
    7: [7, 8],
    8: [10, 11],
}

LINE_TO_LINE = {
    1: [16, 17, 18],
    2: [19, 20, 21],
    3: [22, 23, 24],
    4: [19, 20, 21],
    5: [22, 23, 24],
    6: [13, 14, 15],
    7: [22, 23, 24],
    8: [13, 14, 15],
    9: [16, 17, 18],
    10: [13, 14, 15],
    11: [16, 17, 18],
    12: [19, 20, 21],
    13: [6, 8, 10],
    14: [6, 8, 10],
    15: [6, 8, 10],
    16: [1, 9, 11],
    17: [1, 9, 11],
    18: [1, 9, 11],
    19: [2, 4, 12],
    20: [2, 4, 12],
    21: [2, 4, 12],
    22: [3, 5, 7],
    23: [3, 5, 7],
    24: [3, 5, 7],
}


def load_pickle(file):
    with open(file, "rb") as f:
        output = pickle.load(f)
    return output


def get_action(obs, last_action, hist, change):
    x = [0] * 9
    x[last_action] += max(0.5, change)
    y = [0] * 9

    for i in range(1, 13):
        lane_vehicle_num = obs[i]
        for v in LINE_TO_ACTION[i]:
            x[v] += lane_vehicle_num

        hist_vehiche_num = hist[i]
        for v in LINE_TO_ACTION[i]:
            y[v] += hist_vehiche_num

    x = np.array(x[1:]).reshape(-1)
    y = np.array(y[1:]).reshape(-1)
    z = np.ones(len(x)).reshape(-1) * (-1)
    # print(x)
    actions = np.argwhere(x == np.amax(x)).reshape(-1)
    z[actions] = y[actions]
    action = z.argmax() + 1
    return action


# def get_action(obs, last_action, hist, change):
#     x = [0] * 9
#     x[last_action] += max(0.5, change)
#     y = [0] * 9
#     for i in range(1, 13):
#         lane_vehicle_num = obs[i]
#         for v in LINE_TO_ACTION[i]:
#             x[v] += lane_vehicle_num
#
#         hist_vehiche_num = hist[i]
#         for v in LINE_TO_ACTION[i]:
#             y[v] += hist_vehiche_num
#
#     for i in range(13, 25):
#         lane_vehicle_num = obs[i]
#         for a in LINE_TO_LINE[i]:
#             for v in LINE_TO_ACTION[a]:
#                 x[v] -= lane_vehicle_num * 0.01
#
#     x = np.array(x[1:]).reshape(-1)
#     y = np.array(y[1:]).reshape(-1)
#     z = np.ones(len(x)).reshape(-1) * (-1)
#
#     actions = np.argwhere(x == np.amax(x)).reshape(-1)
#     z[actions] = y[actions]
#     action = z.argmax() + 1
#     return action


class TestAgent:
    def __init__(self):
        self.step_time = 10
        self.red_sec = 5
        self.agent_list = []
        self.last_actions = {}
        self.last_last_actions = {}
        self.actlist = []
        self.roads_importance = defaultdict(lambda: 0)
        self.free_flow_est = defaultdict(lambda: [0, []])
        self.lane_set = set()

    def load_agent_list(self, agent_list):
        for r in ["vehicle", "current_time", "lane_rank"]:
            agent_list.remove(r)
        self.agent_list = agent_list
        self.last_actions = {int(agent): 1 for agent in self.agent_list}

    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

    def act(self, obs):
        infos = obs["observations"]["vehicle"]
        current_time = obs["observations"]["current_time"]
        lane_rank = obs["observations"]["lane_rank"]

        actions = {}
        observations_for_agent = defaultdict(lambda: [0] * 25)
        red_for_agent = defaultdict(lambda: [0] * 25)
        change_observations_for_agent = defaultdict(lambda: [0] * 1)

        for _, info in infos.items():
            start_time = float(info["start_time"][0])
            ff = info["ff"]
            first_route = info["route"][0]
            max_free_flow = self.free_flow_est[(first_route, start_time // 200)][0]
            if ff >= max_free_flow:
                self.free_flow_est[(first_route, start_time // 200)][0] = ff
                self.free_flow_est[(first_route, start_time // 200)][1] = info["route"]

        for vehicle_id, info in infos.items():
            first_route = info["route"][0]
            start_time = float(info["start_time"][0])
            est_ff = self.free_flow_est[(first_route, start_time // 200)][0]
            ff = info["ff"]

            if est_ff > 0:
                delay_index = np.clip(
                    ((current_time - start_time) + (est_ff - ff)) / est_ff, 0, 10
                )
            else:
                delay_index = 1

            w = 1 + (delay_index - 1) * 0.5

            road = self.roads[int(info["road"][0])]
            road["road"] = int(info["road"][0])
            end_inter = road["end_inter"]
            start_time = float(info["start_time"][0])
            drivable = int(info["drivable"][0])  # lane

            if end_inter in self.last_actions:
                last_action = self.last_actions[end_inter]
            else:
                last_action = -1

            current_lane_rank = lane_rank[drivable]
            for idx in range(len(current_lane_rank)):
                if current_lane_rank[idx][3] == vehicle_id:
                    break
            sub_lane = current_lane_rank[idx][4]
            front_vehicle_num = 0

            for i in range(idx + 1, len(current_lane_rank)):
                if current_lane_rank[i][4] == sub_lane:
                    front_vehicle_num += 1

            if front_vehicle_num <= 10:
                current_vehicle_speed = info["speed"][0]
                road_speed_limit = road["speed_limit"]
                rest_time = 0
                acc_time = (road_speed_limit - current_vehicle_speed) / 2.0
                acc_dis = (road_speed_limit + current_vehicle_speed) / 2.0 * acc_time
                distance = info["distance"][0]
                if acc_dis + distance > road["length"]:
                    rest_distance = road["length"] - distance
                    acc_time_finish = (
                        -current_vehicle_speed
                        + np.sqrt(current_vehicle_speed ** 2 + 4 * rest_distance)
                    ) / 2.0
                    rest_time = acc_time_finish
                else:
                    rest_time += acc_time
                    distance += acc_dis
                    rest_distance = road["length"] - distance
                    rest_time += rest_distance / road_speed_limit
            else:
                rest_time = float("inf")

            if self.intersections[end_inter]["have_signal"] != 0:
                lane = (
                    self.intersections[end_inter]["lanes"].index(
                        int(info["drivable"][0])
                    )
                    + 1
                )

                max_route = self.free_flow_est[(info["route"][0], start_time // 200)][1]

                is_in = drivable in max_route
                upcoming_drivable_count = 0
                is_not_last = False

                if is_in:
                    is_not_last = max_route[-1] != drivable

                upcoming_drivables = [
                    int(self.intersections[end_inter]["lanes"][c - 1])
                    for c in LINE_TO_LINE[lane]
                ]

                threshold_adj = 0.2
                if (is_in) and (is_not_last):
                    upcoming_drivable = max_route[max_route.index(drivable) + 1]
                    if upcoming_drivable in upcoming_drivables:
                        upcoming_drivables = [upcoming_drivable]
                        threshold_adj = 0.6 #0.4 #0.5

                for upcoming_drivable in upcoming_drivables:
                    if upcoming_drivable in lane_rank:
                        upcoming_road = self.roads[upcoming_drivable // 100]
                        for k in lane_rank[upcoming_drivable]:
                            threshold = upcoming_road["length"] * threshold_adj
                            if (k[0] <= threshold) and (k[2] < 1):
                                if len(infos[k[3]]['route'])!=1:
                                    #if current_time-infos[k[3]]['start_time']<=10:
                                    #print(upcoming_road['start_inter'])
                                    upcoming_drivable_count += 1
                                    break
                            elif k[0] >= threshold:
                                break

                if upcoming_drivable_count >= 1:
                    w = 0  # 1-upcoming_drivable_count/len(upcoming_drivables)

                w *= 1 - (front_vehicle_num / 10) * 0.5

                if lane not in [3, 6, 9, 12]:
                    if last_action != -1:
                        is_open = lane in ACTION_TO_LINE[last_action]
                    else:
                        is_open = False

                    if is_open:
                        if rest_time <= self.step_time:
                            observations_for_agent[end_inter][lane] += 1 * w
                            if w > 0:
                                change_observations_for_agent[end_inter][0] += 0.25
                        else:
                            if w > 0:
                                red_for_agent[end_inter][lane] += 1
                    else:

                        if self.step_time - self.red_sec <= rest_time <= self.step_time:
                            observations_for_agent[end_inter][lane] += 0.5 * w
                            if w > 0:
                                change_observations_for_agent[end_inter][0] += 0.25

                        elif rest_time < self.step_time - self.red_sec:
                            observations_for_agent[end_inter][lane] += 1 * w

                        else:
                            if w > 0:
                                red_for_agent[end_inter][lane] += 1

                else:
                    if rest_time < self.step_time - self.red_sec:
                        if w > 0:
                            change_observations_for_agent[end_inter][0] += 0.25

        for agent in self.agent_list:
            agent = int(agent)

            actions[agent] = get_action(
                observations_for_agent[agent],
                self.last_actions[agent],
                red_for_agent[agent],
                change_observations_for_agent[agent][0],
            )

        self.last_actions = actions.copy()
        return actions


scenario_dirs = ["test"]
agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    agent_specs[k] = TestAgent()
