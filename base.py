

import sys
import os
import threading
from tempfile import TemporaryDirectory
from typing import Dict, Any, Tuple, Optional, List
from queue import Queue, Empty, Full
from datetime import datetime
import math
import time

import random

# import scipy
# import pickle

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box, MultiDiscrete

sys.path.insert(0, '/home/ck/Downloads/EnergyPlus-23.1.0-87ed9199d4-Linux-CentOS7.9.2009-x86_64/')
from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.datatransfer import DataExchange


class EnergyPlusRunner:

    def __init__(self, episode: int, env_config: Dict[str, Any], obs_queue: Queue, act_queue: Queue, meter_queue: Queue) -> None:
        self.episode = episode
        self.env_config = env_config
        self.meter_queue = meter_queue
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.curr = None # current time of the simulation run for output directory naming

        self.energyplus_api = EnergyPlusAPI()
        self.x: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized = False
        self.init_queue = Queue()
        self.progress_value: int = 0
        self.simulation_complete = False
        self.request_variable_complete = False



        # below is declaration of variables, meters and actuators
        # this simulation will interact with
        self.variables = {
            'var-attic-indoor_temperature': ('Zone Air Temperature', 'Attic'),
            'var-core_zn-indoor_temperature': ('Zone Air Temperature', 'Core_ZN'),
            'var-perimeter_zn_1-indoor_temperature': ('Zone Air Temperature', 'Perimeter_ZN_1'),
            'var-perimeter_zn_2-indoor_temperature': ('Zone Air Temperature', 'Perimeter_ZN_2'),
            'var-perimeter_zn_3-indoor_temperature': ('Zone Air Temperature', 'Perimeter_ZN_3'),
            'var-perimeter_zn_4-indoor_temperature': ('Zone Air Temperature', 'Perimeter_ZN_4'),
            'var_environment_site_outdoor_air_drybulb_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
            'var_environment_site_direct_solar_radiation_rate_per_area': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
            'var_environment_site_horizontal_infrared_radiation_rate_per_area': ('Site Horizontal Infrared Radiation Rate per Area', 'Environment'),
            'var_environment_site_diffuse_solar_radiation_rate_per_area': ('Site Diffuse Solar Radiation Rate per Area', 'Environment')
        }


        self.var_handles: Dict[str, int] = {}
        self.var_handles_index: Dict[str, int] = {}
        for i in range(len(self.variables)):
            self.var_handles_index[list(self.variables.keys())[i]] = i

        self.meters = {
            "elec_hvac": "Electricity:HVAC",
            "elec_heating": "Heating:Electricity",
            "elec_cooling": "Cooling:Electricity",
            'elec_facility': "Electricity:Facility",

            'cooling_core_zn': 'Cooling:EnergyTransfer:Zone:CORE_ZN',
            'cooling_perimeter_1': 'Cooling:EnergyTransfer:Zone:PERIMETER_ZN_1',
            'cooling_perimeter_2': 'Cooling:EnergyTransfer:Zone:PERIMETER_ZN_2',
            'cooling_perimeter_3': 'Cooling:EnergyTransfer:Zone:PERIMETER_ZN_3',
            'cooling_perimeter_4': 'Cooling:EnergyTransfer:Zone:PERIMETER_ZN_4',
            # 'core_elec': 'Electricity:Core_ZN:HVAC'
        }
        self.meter_handles: Dict[str, int] = {}

        self.actuators = {
            "cooling_actuator_Core_ZN" : (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "Core_ZN"
            ),
            "cooling_actuator_Perimeter_ZN_1" : (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "Perimeter_ZN_1"
            ),
            "cooling_actuator_Perimeter_ZN_2" : (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "Perimeter_ZN_2"
            ),
            "cooling_actuator_Perimeter_ZN_3" : (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "Perimeter_ZN_3"
            ),
            "cooling_actuator_Perimeter_ZN_4" : (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "Perimeter_ZN_4"
            ),

            "heating_actuator_Core_ZN" : (
                "Zone Temperature Control",
                "Heating Setpoint",
                "Core_ZN"
            ),
            "heating_actuator_Perimeter_ZN_1" : (
                "Zone Temperature Control",
                "Heating Setpoint",
                "Perimeter_ZN_1"
            ),
            "heating_actuator_Perimeter_ZN_2" : (
                "Zone Temperature Control",
                "Heating Setpoint",
                "Perimeter_ZN_2"
            ),
            "heating_actuator_Perimeter_ZN_3" : (
                "Zone Temperature Control",
                "Heating Setpoint",
                "Perimeter_ZN_3"
            ),
            "heating_actuator_Perimeter_ZN_4" : (
                "Zone Temperature Control",
                "Heating Setpoint",
                "Perimeter_ZN_4"
            ),
        }
        self.actuator_handles: Dict[str, int] = {}

    def start(self) -> None:
        self.energyplus_state = self.energyplus_api.state_manager.new_state()
        runtime = self.energyplus_api.runtime

        # requesting variables to E+ to be avaiable during runtime
        if not self.request_variable_complete:
            for key, var in self.variables.items():
                self.x.request_variable(self.energyplus_state, var[0], var[1])
                self.request_variable_complete = True

        # register callback used to track simulation progress
        def report_progress(progress: int) -> None:
            self.progress_value = progress

        runtime.callback_progress(self.energyplus_state, report_progress)

        # register callback used to collect observations
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_meter)

        # register callback used to send actions
        runtime.callback_after_predictor_after_hvac_managers(self.energyplus_state, self._send_actions)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(runtime, cmd_args, state, results):
            print(f"running EnergyPlus with args: {cmd_args}")

            # start simulation
            results["exit_code"] = runtime.run_energyplus(state, cmd_args)

        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(
                self.energyplus_api.runtime,
                self.make_eplus_args(),
                self.energyplus_state,
                self.sim_results
            )
        )
        self.energyplus_exec_thread.start()

    def stop(self) -> None:
        if self.energyplus_exec_thread:
            self.simulation_complete = True
            self._flush_queues()
            self.energyplus_exec_thread.join()
            self.energyplus_exec_thread = None
            self.energyplus_api.runtime.clear_callbacks()
            self.energyplus_api.state_manager.delete_state(self.energyplus_state)

    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        """
        make command line arguments to pass to EnergyPlus
        """
        self.curr = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        eplus_args = ["-r"] if self.env_config.get("csv", False) else []
        eplus_args += ['-x']

        eplus_args += ["-a"] if self.env_config.get('annual', False) else []
        eplus_args += [
            "-w",
            self.env_config["epw"],
            "-d",
            # change below for output directory name formatting
            f"{self.env_config['output']}/episode-{self.episode}-{datetime.now()}",
            # f"{self.env_config['output']}/episode-{self.episode:08}-{os.getpid():05}",
            self.env_config["idf"]
        ]
        print(eplus_args)
        return eplus_args

    def _collect_meter(self, state_argument) -> None:
        '''
        For addressing -> values used to calculate rewards also seen in observation of the agent
        '''
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        self.next_meter = {
            **{
                key: self.x.get_meter_value(state_argument, handle)
                for key, handle
                in self.meter_handles.items()
            }
        }
        self.meter_queue.put(self.next_meter)

    def _collect_obs(self, state_argument) -> None:
        """
        EnergyPlus callback that collects output variables/meters
        values and enqueue them
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            # print('HIT COLLECT OBS')
            return

        self.next_obs = {
            **{
                key: self.x.get_variable_value(state_argument, handle)
                for key, handle
                in self.var_handles.items()
            },
        }

        year = self.x.year(self.energyplus_state)
        month = self.x.month(self.energyplus_state)
        day = self.x.day_of_month(self.energyplus_state)
        hour = self.x.hour(self.energyplus_state)
        minute = self.x.minutes(self.energyplus_state)
        day_of_week = self.x.day_of_week(self.energyplus_state)

        self.next_obs['var-environment-time-month'] = month
        self.next_obs['var-environment-time-day'] = day
        self.next_obs['var-environment-time-hour'] = hour
        self.next_obs['var-environment-time-day_of_week'] = day_of_week

        self.obs_queue.put(self.next_obs)

    def _continuous_rescale(self, action, old_range_min, old_range_max, new_range_min, new_range_max):
        '''
        _rescale method can be used for larger range to smaller range
        '''
        old_range = old_range_max - old_range_min
        new_range = new_range_max - new_range_min
        return (((action - old_range_min) * new_range) / old_range) + new_range_min

    def _discrete_rescale(self, action, number_of_discrete, new_range_min, new_range_max):
        action_nparray = np.linspace(new_range_min, new_range_max, number_of_discrete)
        return action_nparray[int(action)]

    def _send_actions(self, state_argument):
        """
        EnergyPlus callback that sets actuator value from last decided action
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        if self.act_queue.empty():
            return
        temp_next_action = self.act_queue.get()
        #next_action = self._rescale(next_action, -1, 1, 15, 30)

        next_action = []
        for action_val in temp_next_action:
            rescaled_action = self._discrete_rescale(action_val, 61, 20, 26)
            next_action.append(rescaled_action)
            #next_action.append(np.float32(self._discrete_rescale(action, 61, 20, 26)))

        #assert all(isinstance(action_val, float) or isinstance(action_val, np.float32) for action_val in next_action)

        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['cooling_actuator_Core_ZN'],
            actuator_value=next_action[0]
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['cooling_actuator_Perimeter_ZN_1'],
            actuator_value=next_action[1]
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['cooling_actuator_Perimeter_ZN_2'],
            actuator_value=next_action[2]
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['cooling_actuator_Perimeter_ZN_3'],
            actuator_value=next_action[3]
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['cooling_actuator_Perimeter_ZN_4'],
            actuator_value=next_action[4]
        )

        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['heating_actuator_Core_ZN'],
            actuator_value=0 # NOTE: set it to a extreme low temp so it's never triggered
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['heating_actuator_Perimeter_ZN_1'],
            actuator_value=0 # NOTE: set it to a extreme low temp so it's never triggered
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['heating_actuator_Perimeter_ZN_2'],
            actuator_value=0 # NOTE: set it to a extreme low temp so it's never triggered
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['heating_actuator_Perimeter_ZN_3'],
            actuator_value=0 # NOTE: set it to a extreme low temp so it's never triggered
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['heating_actuator_Perimeter_ZN_4'],
            actuator_value=0 # NOTE: set it to a extreme low temp so it's never triggered
        )

    def _init_callback(self, state_argument) -> bool:
        """initialize EnergyPlus handles and checks if simulation runtime is ready"""
        self.initialized = self._init_handles(state_argument) and not self.x.warmup_flag(state_argument)
        return self.initialized

    #  DONE NOTE: some error with multiple request of handles -> WARNINGS for now but good to fix
    def _init_handles(self, state_argument):
        """initialize sensors/actuators handles to interact with during simulation"""
        if not self.initialized:
            if not self.x.api_data_fully_ready(state_argument):
                return False


            self.var_handles = {
                key: self.x.get_variable_handle(state_argument, *var)
                for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: self.x.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: self.x.get_actuator_handle(state_argument, *actuator)
                for key, actuator in self.actuators.items()
            }

            for handles in [
                    self.var_handles,
                    self.meter_handles,
                    self.actuator_handles
            ]:
                if any([v == -1 for v in handles.values()]):
                    available_data = self.x.list_available_api_data_csv(state_argument).decode('utf-8')
                    print(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}"
                        # NOTE: commented out for now
                    )
                    exit(1)

            self.init_queue.put("")
            self.initialized = True

        return True

    def _flush_queues(self):
        for q in [self.obs_queue, self.act_queue]:
            while not q.empty():
                q.get()




class EnergyPlusEnv(gym.Env):

    def __init__(self, env_config: Dict[str, Any]):
        self.env_config = env_config
        self.episode = -1
        self.timestep = 0
        # self.start_date = datetime(2000, env_config['start_date'][0], env_config['start_date'][1])
        # self.end_date = datetime(2000, env_config['end_date'][0], env_config['end_date'][1])

        obs_len = 10
        low_obs = np.array(
            [-100.0, -100.0, -100.0, 0, 0, 0, 0, 0]
        )
        hig_obs = np.array(
            [100.0, 100.0, 100.0, 100.0, 100000000.0, 100000000.0, 100000000.0, 100000000.0]
        )
        self.observation_space = gym.spaces.Box(
            low=low_obs, high=hig_obs, dtype=np.float64
            # dtype was originally set to float32
        )
        self.last_obs = {}

        # last obs saves it in dictionary format, but last_next_state saves the numpy vec
        self.last_next_state = None

        self.prev_obs = None

        self.action_space: MultiDiscrete = MultiDiscrete([61, 61, 61, 61, 61])

        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.meter_queue: Optional[Queue] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None

    def _rescale(self, action, old_range_min, old_range_max, new_range_min, new_range_max):
        '''
        _rescale already implemented for EnergyPlusRunner class, but for convenience, implemented
        for EnergyPlusEnv
        '''
        old_range = old_range_max - old_range_min
        new_range = new_range_max - new_range_min
        return (((action - old_range_min) * new_range) / old_range) + new_range_min



    def retrieve_actuators(self):
        '''
        for debugging purposes: fetches actuator values (cooling, heating)
        '''
        runner = self.energyplus_runner
        cooling_actuator_Core_ZN_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['cooling_actuator_Core_ZN'])
        cooling_actuator_Perimeter_ZN_1_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['cooling_actuator_Perimeter_ZN_1'])
        cooling_actuator_Perimeter_ZN_2_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['cooling_actuator_Perimeter_ZN_2'])
        cooling_actuator_Perimeter_ZN_3_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['cooling_actuator_Perimeter_ZN_3'])
        cooling_actuator_Perimeter_ZN_4_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['cooling_actuator_Perimeter_ZN_4'])
        heating_actuator_Core_ZN_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['heating_actuator_Core_ZN'])
        heating_actuator_Perimeter_ZN_1_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['heating_actuator_Perimeter_ZN_1'])
        heating_actuator_Perimeter_ZN_2_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['heating_actuator_Perimeter_ZN_2'])
        heating_actuator_Perimeter_ZN_3_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['heating_actuator_Perimeter_ZN_3'])
        heating_actuator_Perimeter_ZN_4_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['heating_actuator_Perimeter_ZN_4'])
        cooling_values = [cooling_actuator_Core_ZN_value, cooling_actuator_Perimeter_ZN_1_value, cooling_actuator_Perimeter_ZN_2_value, cooling_actuator_Perimeter_ZN_3_value, cooling_actuator_Perimeter_ZN_4_value]
        heating_values = [heating_actuator_Core_ZN_value, heating_actuator_Perimeter_ZN_1_value, heating_actuator_Perimeter_ZN_2_value, heating_actuator_Perimeter_ZN_3_value, heating_actuator_Perimeter_ZN_4_value]
        return cooling_values, heating_values

    def reset(
            self, *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ):
        self.episode += 1
        self.last_obs = self.observation_space.sample()

        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

        # observations and actions queues for flow control
        # queues have a default max size of 1
        # as only 1 E+ timestep is processed at a time
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)
        self.meter_queue = Queue(maxsize=1)

        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            env_config=self.env_config,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
            meter_queue=self.meter_queue
        )
        self.energyplus_runner.start()

        # wait for E+ warmup to complete
        if not self.energyplus_runner.initialized:
            self.energyplus_runner.init_queue.get()

        try:
            obs = self.obs_queue.get()
            meter = self.meter_queue.get()
        except Empty:
            meter = self.last_meter
            obs = self.last_obs

        obs_vec = np.array(list(obs.values()))

        self.last_next_state = obs_vec

        return obs_vec

    def step(self, action):
        '''
        @params: action -> numpy.ndarray w/ 1 element
        '''
        self.timestep += 1
        done = False

        # check for simulation errors
        if self.energyplus_runner.failed():
            print(f"EnergyPlus failed with {self.energyplus_runner.sim_results['exit_code']}")
            exit(1)

        #action = np.float32(action)
        action = np.array([np.float32(action_val) for action_val in action])

        # enqueue action (received by EnergyPlus through dedicated callback)
        # then wait to get next observation.
        # timeout is set to 2s to handle start and end of simulation cases, which happens async
        # and materializes by worker thread waiting on this queue (EnergyPlus callback
        # not consuming yet/anymore)
        # timeout value can be increased if E+ warmup period is longer or if step takes longer
        timeout = 2
        try:
            self.act_queue.put(action, timeout=timeout)
            self.last_obs = obs = self.obs_queue.get(timeout=timeout)
            self.last_meter = meter = self.meter_queue.get(timeout=timeout)
        except (Full, Empty):
            done = True
            obs = self.last_obs
            meter = self.last_meter


        obs_vec = np.array(list(obs.values()))


        #print('METER:', meter)

        # update the self.last_next_state
        self.last_next_state = obs_vec

        # compute energy reward
        reward_energy = self._compute_reward_energy(meter)
        reward_zone_cooling_energy_transfer = self._compute_zone_energy_transfer(meter)


        # NOTE: changed this to 99 but 100 works fine also
        if self.energyplus_runner.progress_value == 99:
            print("reached end of simulation")
            done = True


        return obs_vec, reward_zone_cooling_energy_transfer, done, False, {'cooling_core': meter['cooling_core_zn'],
                                                                           'cooling_perimeter_1': meter['cooling_perimeter_1'],
                                                                           'cooling_perimeter_2': meter['cooling_perimeter_2'],
                                                                           'cooling_perimeter_3': meter['cooling_perimeter_3'],
                                                                           'cooling_perimeter_4': meter['cooling_perimeter_4']}

    def render(self, mode="human"):
        # TODO? : maybe add IDF visualization option
        pass

    @staticmethod
    def _compute_reward_thermal_comfort(tdb, tr, v, rh) -> float:
        '''
        @params
        tdb: dry bulb air temperature
        tr: mean radiant temperature
        v: used to calculate v_relative: air velocity
        rh: relative humidity
        met: set as a constant value of 1.4
        clo: set as a constant value of 0.5
        -> clo_relative is pre-computed ->

        @return PPD
        '''
        def pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme):
            pa = rh * 10 * math.exp(16.6536 - 4030.183 / (tdb + 235))

            icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
            m = met * 58.15  # metabolic rate in W/M2
            w = wme * 58.15  # external work in W/M2
            mw = m - w  # internal heat production in the human body
            # calculation of the clothing area factor
            if icl <= 0.078:
                f_cl = 1 + (1.29 * icl)  # ratio of surface clothed body over nude body
            else:
                f_cl = 1.05 + (0.645 * icl)

            # heat transfer coefficient by forced convection
            hcf = 12.1 * math.sqrt(vr)
            hc = hcf  # initialize variable
            taa = tdb + 273
            tra = tr + 273
            t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

            p1 = icl * f_cl
            p2 = p1 * 3.96
            p3 = p1 * 100
            p4 = p1 * taa
            p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
            xn = t_cla / 100
            xf = t_cla / 50
            eps = 0.00015

            n = 0
            while abs(xn - xf) > eps:
                xf = (xf + xn) / 2
                hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
                if hcf > hcn:
                    hc = hcf
                else:
                    hc = hcn
                    xn = (p5 + p4 * hc - p2 * xf**4) / (100 + p3 * hc)
                    n += 1
                if n > 150:
                    raise StopIteration("Max iterations exceeded")

            tcl = 100 * xn - 273

            # heat loss diff. through skin
            hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
            # heat loss by sweating
            if mw > 58.15:
                hl2 = 0.42 * (mw - 58.15)
            else:
                hl2 = 0
                # latent respiration heat loss
            hl3 = 1.7 * 0.00001 * m * (5867 - pa)
            # dry respiration heat loss
            hl4 = 0.0014 * m * (34 - tdb)
            # heat loss by radiation
            hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
            # heat loss by convection
            hl6 = f_cl * hc * (tcl - tdb)

            ts = 0.303 * math.exp(-0.036 * m) + 0.028
            _pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

            return _pmv


        def v_relative(v, met):
            """Estimates the relative air speed which combines the average air speed of
            the space plus the relative air speed caused by the body movement. Vag is assumed to
            be 0 for metabolic rates equal and lower than 1 met and otherwise equal to
            Vag = 0.3 (M – 1) (m/s)

            Parameters
            ----------
            v : float or array-like
            air speed measured by the sensor, [m/s]
            met : float
            metabolic rate, [met]

            Returns
            -------
            vr  : float or array-like
            relative air speed, [m/s]
            """
            return np.where(met > 1, np.around(v + 0.3 * (met - 1), 3), v)

        def clo_dynamic(clo, met, standard="ASHRAE"):
            """Estimates the dynamic clothing insulation of a moving occupant. The activity as
            well as the air speed modify the insulation characteristics of the clothing and the
            adjacent air layer. Consequently, the ISO 7730 states that the clothing insulation
            shall be corrected [2]_. The ASHRAE 55 Standard corrects for the effect
            of the body movement for met equal or higher than 1.2 met using the equation
            clo = Icl × (0.6 + 0.4/met)

            Parameters
            ----------
            clo : float or array-like
            clothing insulation, [clo]
            met : float or array-like
            metabolic rate, [met]
            standard: str (default="ASHRAE")
            - If "ASHRAE", uses Equation provided in Section 5.2.2.2 of ASHRAE 55 2020

            Returns
            -------
            clo : float or array-like
            dynamic clothing insulation, [clo]
            """
            standard = standard.lower()
            if standard not in ["ashrae", "iso"]:
                raise ValueError(
                    "only the ISO 7730 and ASHRAE 55 2020 models have been implemented"
                )

            if standard == "ashrae":
                return np.where(met > 1.2, np.around(clo * (0.6 + 0.4 / met), 3), clo)
            else:
                return np.where(met > 1, np.around(clo * (0.6 + 0.4 / met), 3), clo)


        clo = 0.5 # precomputed with the clo value of 0.5 (clo_dynamic(0.5, 1.4))
        #v_rel = v_relative(v, 1.4)
        pmv = pmv_ppd_optimized(tdb, tr, 0.1, rh, 1.4, clo, 0)
        return pmv
    #return 100.0 - 95.0 * np.exp(-0.03353 * np.power(pmv, 4.0) - 0.2179 * np.power(pmv, 2.0)) # NOTE: for PPD

    @staticmethod
    def _compute_reward_energy(meter: Dict[str, float]) -> float:
        """compute reward scalar"""
        reward = -1 * meter['elec_cooling']
        return reward

    @staticmethod
    def _compute_zone_energy_transfer(meter: Dict[str, float]) -> float:
        '''compute cooling energy transfer for each zone'''
        core_zn_ret = -1 * meter['cooling_core_zn']
        perimeter_1_ret = -1 * meter['cooling_perimeter_1']
        perimeter_2_ret = -1 * meter['cooling_perimeter_2']
        perimeter_3_ret = -1 * meter['cooling_perimeter_3']
        perimeter_4_ret = -1 * meter['cooling_perimeter_4']
        return [core_zn_ret, perimeter_1_ret, perimeter_2_ret, perimeter_3_ret, perimeter_4_ret]


# NOTE: have to give in -x flag
# default_args = {'idf': '/home/ck/Downloads/Files/in.idf',
#                 'epw': '/home/ck/Downloads/Files/weather.epw',
#                 'csv': True,
#                 'output': './output',
#                 'timesteps': 1000000.0,
#                 'num_workers': 2
#                 }

default_args = {'idf': './in.idf',
                'epw': './weather.epw',
                'csv': False,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                }


if __name__ == "__main__":
    env = EnergyPlusEnv(default_args)
    print('action_space:', end='')
    print(env.action_space)
    print("OBS SHAPE:", env.observation_space.shape)
    scores = []
    for episode in range(1):
        state = env.reset()
        done = False
        score = 0

        test = 0

        while not done:
            action = env.action_space.sample()
            action = [30,30,30,30,30]
            # print('action:', action)
            # print('actuators:', env.retrieve_actuators())
            ret = n_state, reward, done, truncated, info = env.step(action)

            test += info['cooling_core'] + info['cooling_perimeter_1'] + info['cooling_perimeter_2'] + info['cooling_perimeter_3'] + info['cooling_perimeter_4']

            score += reward

        scores.append(score)
        print("SCORES: ", scores)
        print('TEST:', test)

# SCORES:  [-2767056462.5814853]
# SCORE-3887297634.02748]
# SCORES:  [-1794684019.8578289]
# TEST: 11082714705.507038
# TEST: 8080984190.156531
# TEST: 5482413777.092008

# SCORES:  [-5791563413.50407]
# SCORES:  [-10749664786.402805]
