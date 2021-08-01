# -*- coding: utf-8 -*-
import numpy as np
from CBEngine_rllib.CBEngine_rllib import CBEngine_rllib as CBEngine_rllib_class
from collections import defaultdict, namedtuple
import citypb
import os
import pickle

OBS_THRESHOLD = 5


def one_hot(label: int, num_classes: int) -> list:
    output = [0] * num_classes
    output[label] = 1
    return output


def post_process_obs(observations) -> dict:
    keys = list(observations.keys())
    for k in keys:
        observations[str(k)] = {
            "observation": np.array(observations[k]).astype(np.float32)
        }
        observations.pop(k)
    return observations


class CBEngine_round3(CBEngine_rllib_class):
    def __init__(self, config):
        super(CBEngine_round3, self).__init__(config)
        self.observation_features = self.gym_dict["observation_features"]
        self.custom_observation = self.gym_dict["custom_observation"]
        self.observation_dimension = self.gym_dict["observation_dimension"]

    def reset(self):
        del self.eng
        self.eng = citypb.Engine(self.simulator_cfg_file, self.thread_num)
        self.now_step = 0
        self.vehicles.clear()
        self.init_vehicles_route()
        obs = self._get_observations()
        return obs

    def _get_observations(self) -> dict:

        if self.custom_observation == False:
            observations = super(CBEngine_round3, self)._get_observations()
            return observations
        else:
            # choose not to use defaultdict
            observations = {}

            observations["vehicle"] = self._get_vehicle_info()
            observations["current_time"] = self.eng.get_current_time()
            observations["lane_rank"] = self._get_lane_rank()
            observations.update({k: 0 for k in self.agent_signals.keys()})
            # save all vehicle infos
            # self.vehicle_infos = self._get_vehicle_infos()

            # self.observations = post_process_obs(self.observations)
            return observations

    # output dim: 8
    def get_current_phase(self) -> None:
        self.history_agent_phase = self.agent_curphase.copy()
        for k, v in self.history_agent_phase.items():
            self.observations[k].extend(one_hot(v - 1, 8))

    def init_vehicles_route(self) -> None:
        self.vehicles_route = defaultdict(lambda: [0, 0, []])

    def _get_est_ff(self, vehicle_id):
        vehicle_info = self.eng.get_vehicle_info(vehicle_id)
        current_road = int(vehicle_info["drivable"][0])
        distance = float(vehicle_info["distance"][0])
        last_road = self.vehicles_route[vehicle_id][0]

        if last_road != current_road:
            self.vehicles_route[vehicle_id][0] = current_road
            self.vehicles_route[vehicle_id][2].append(current_road)

            if last_road != 0:
                self.vehicles_route[vehicle_id][1] += (
                    self.roads[last_road // 100]["length"]
                    / self.roads[last_road // 100]["speed_limit"]
                )

        est_ff = (
            self.vehicles_route[vehicle_id][1]
            + distance / self.roads[current_road // 100]["speed_limit"]
        )

        is_initial_vehicle = last_road == 0
        return vehicle_info, est_ff, is_initial_vehicle

    def _get_info(self):
        infos = {}
        return infos

    def _get_vehicle_info(self):
        infos = {}
        current_time = self.eng.get_current_time()
        v_list = self.eng.get_vehicles()
        for vehicle in v_list:
            info, est_ff, is_initial = self._get_est_ff(vehicle)
            infos[vehicle] = info
            start_time = float(info["start_time"][0])
            if est_ff > 0:
                delay_index = (current_time - start_time) / est_ff
                delay_index = np.clip(delay_index, 0, 100)
            else:
                delay_index = 1
            infos[vehicle]["delay_index"] = delay_index
            infos[vehicle]["ff"] = est_ff
            infos[vehicle]["route"] = self.vehicles_route[vehicle][2]
        return infos

    def _get_lane_rank(self):
        lane_vehicle = defaultdict(lambda: [])
        vehicles = self.eng.get_vehicles()
        for vehicle_id in vehicles:
            info = self.eng.get_vehicle_info(vehicle_id)
            lane_id = int(info["drivable"][0])
            distance = info["distance"][0]
            remain_distance = self.roads[lane_id // 100]["length"] - float(distance)
            speed = int(info["speed"][0])
            lane_vehicle[lane_id].append([distance, remain_distance, speed, vehicle_id])

        for k in lane_vehicle:
            lane_vehicle[k].sort(key=lambda x: x[0])
            len_lane_vehicle = len(lane_vehicle[k])
            left_last = len_lane_vehicle - 1
            right_last = None
            lane_vehicle[k][left_last].append(0)

            for idx in range(len_lane_vehicle - 2, -1, -1):
                distance, remain_distance, speed, vehicle_id = lane_vehicle[k][idx]
                (
                    left_distance,
                    left_remain_distance,
                    left_speed,
                    left_vehicle_id,
                    _,
                ) = lane_vehicle[k][left_last]

                if right_last is None:
                    if (distance - left_distance) <= 5:
                        lane_vehicle[k][idx].append(1)
                        # right_last = idx
                    else:
                        lane_vehicle[k][idx].append(0)
                        left_last = idx
                else:
                    (
                        right_distance,
                        right_remain_distance,
                        right_speed,
                        right_vehicle_id,
                        _,
                    ) = lane_vehicle[k][right_last]
                    if ((distance - left_distance) <= 5) and (
                        (distance - right_distance) > 5
                    ):
                        lane_vehicle[k][idx].append(1)
                        right_last = idx
                    elif ((distance - left_distance) > 5) and (
                        (distance - right_distance) <= 5
                    ):
                        lane_vehicle[k][idx].append(0)
                        left_last = idx
                    # elif (distance - left_distance) <= (distance - right_distance):
                    #     lane_vehicle[k][idx].append(0)
                    #     left_last = idx
                    else:
                        # lane_vehicle[k][idx].append(1)
                        # right_last = idx
                        lane_vehicle[k][idx].append(0)
                        left_last = idx
        return lane_vehicle
        # for k in lane_vehicle:
        #     is_left_max = False
        #     is_right_max = False
        #     left_count = 0
        #     right_count = 0
        #     for v in lane_vehicle[k][::-1]:
        #
        #         info = self.eng.get_vehicle_info(v[3])
        #         remain_time = self.get_remain_time(info)  # + extra_time
        #         v.append(remain_time)
        #         if v[4] == 0:
        #             v.append(left_count)
        #             left_count += 1
        #             if is_left_max:
        #                 v.append(0)
        #             else:
        #                 if remain_time <= 10:
        #                     v.append(1)
        #                 else:
        #                     v.append(0)
        #                     is_left_max = True
        #         else:
        #             v.append(right_count)
        #             right_count += 1
        #             if is_right_max:
        #                 v.append(0)
        #             else:
        #                 if remain_time <= 10:
        #                     v.append(1)
        #                 else:
        #                     v.append(0)
        #                     is_right_max = True
        #
        # lane_vehicle = dict(lane_vehicle)

    # def get_remain_time(self, info):
    #     remain_time = 0
    #     speed = info["speed"][0]
    #     distance = info["distance"][0]  # - extra_dis
    #     road = info["road"][0]
    #     length = self.roads[road]["length"]
    #     speed_limit = self.roads[road]["speed_limit"]
    #     acc_time = (speed_limit - speed) / 2.0
    #     acc_dis = (speed_limit + speed) / 2.0 * acc_time
    #     if acc_dis + distance > length:
    #         remain_dis = length - distance
    #         acc_time_finish = (-speed + np.sqrt(speed * speed + 4 * remain_dis)) / 2.0
    #         remain_time += acc_time_finish
    #     else:
    #         remain_time += acc_time
    #         distance += acc_dis
    #         remain_dis = length - distance
    #         remain_time += remain_dis / speed_limit
    #     return remain_time

    def _get_reward(self) -> dict:
        rewards = {}
        return rewards
