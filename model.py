import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler
from agents import Citizen, Cop
from utils import CitizenActions, CopActions, CitizenMemoryInitial, CopMemoryInitial
import names_generator


class Corruption(Model):
    def __init__(self,
                 num_citizens=500,
                 num_cops=100,
                 team_size=10,
                 rationality_of_agents=10,  # 0 is random totally
                 jail_time=4,
                 prob_of_prosecution=0.7,  # ground truth
                 memory_size=10,
                 fine_amount=1.,  # don't change this in sensitivity analysis
                 cost_complain=0.4,
                 penalty_citizen_prosecution=0.,
                 jail_cost_factor=0.5,
                 # jail cost and jail_time should somehow relate to each other I think, but don't know how exactly
                 cost_accept_mean_std=(0.2, 0.05),
                 citizen_complain_memory_discount_factor=0.5,
                 bribe_amount=0.5,
                 moral_commitment_mean_std=(0.25, 0.1),
                 initial_time_left_in_jail=0,  # don't think it's worth to change that
                 initial_indifferent_corruption_honest_rate=(1.0, 0.0, 0.0),
                 corruption_among_teams_spread=1.0,
                 # rate of teams that should be getting the corrupted cops. 1 - all teams have the same amount(+-1 cop ofc)
                 logger: bool = True
                 ):

        super().__init__()

        now = datetime.now()  # current date and time
        if logger:
            self.experiment_name = names_generator.generate_name() + "_" + now.strftime("%d_%m_%H_%M")
            print("Experiment name: ", self.experiment_name)

        # saving everything, then it can be logged
        self.bribe_amount = bribe_amount
        self.initial_time_left_in_jail = initial_time_left_in_jail
        self.cost_accept_mean_std = cost_accept_mean_std
        self.moral_commitment_mean_std = moral_commitment_mean_std
        self.citizen_complain_memory_discount_factor = citizen_complain_memory_discount_factor
        self.prob_of_prosecution = prob_of_prosecution
        self.memory_size = memory_size
        self.cost_complain = cost_complain
        self.penalty_citizen_prosecution = penalty_citizen_prosecution
        self.fine_amount = fine_amount
        self.rationality_of_agents = rationality_of_agents
        self.num_citizens = num_citizens
        self.num_cops = num_cops
        assert self.num_cops <= self.num_citizens, "There should be more citizens than cops!"
        # cops calculations
        self.num_indifferent_cops = int(initial_indifferent_corruption_honest_rate[0] * num_cops)
        self.num_corrupted_cops = int(initial_indifferent_corruption_honest_rate[1] * num_cops)
        self.num_honest_cops = int(initial_indifferent_corruption_honest_rate[2] * num_cops)
        self.num_honest_cops += self.num_cops - (self.num_corrupted_cops + self.num_honest_cops + self.num_honest_cops)
        # citizen calculations
        self.num_indifferent_citizens = int(initial_indifferent_corruption_honest_rate[0] * num_citizens)
        self.num_corrupted_citizens = int(initial_indifferent_corruption_honest_rate[1] * num_citizens)
        self.num_honest_citizens = int(initial_indifferent_corruption_honest_rate[2] * num_citizens)
        self.num_honest_citizens += self.num_citizens - (
                self.num_corrupted_citizens + self.num_honest_citizens + self.num_honest_citizens)

        self.team_size = team_size
        assert self.num_cops % self.team_size == 0, \
            f"You need to set num of cops to be dividable by team size. Each team should have the same size! Your num_cops: {self.num_cops}, teamsize: {self.team_size}"
        self.number_of_teams = math.ceil(self.num_cops / self.team_size)

        # how many iterations cop is inactive
        self.jail_time = jail_time
        # actual cost that cop takes into consideration in the utility function
        self.jail_cost_factor = jail_cost_factor
        self.jail_cost = self.jail_cost_factor * jail_time

        assert sum(
            [rate for rate in initial_indifferent_corruption_honest_rate]) == 1.0, "Distribution should sum up to 1."

        # Checking if the value isn't to small, if it is set it to minimum
        self.corruption_among_teams_spread = corruption_among_teams_spread
        self.number_of_corrupted_teams = math.ceil(max(corruption_among_teams_spread * self.number_of_teams,
                                                       initial_indifferent_corruption_honest_rate[
                                                           1] * num_cops / self.team_size))
        # Initialise schedulers
        self.schedule = BaseScheduler(self)
        self.schedule_Cop = BaseScheduler(self)
        self.init_agents()
        # Data collector to be able to save the data
        self.datacollector = self.get_server_data_collector()

        # Divide the cops over a network of teams
        self.create_network()

        self.logger = logger
        if self.logger:
            # Should be after all initializations as it saves all params!
            self.init_logger()

    def step(self):
        self.cops_playing = [cop for cop in self.schedule_Cop.agents if cop.time_left_in_jail == 0]
        self.citizens_playing = random.sample(self.schedule.agents, len(self.cops_playing))

        self.schedule.step()
        self.schedule_Cop.step()

        self.datacollector.collect(self)
        self.update_network()
        if self.logger:
            self.log_data(self.schedule.steps)

    def init_agents(self):
        # Add agents to schedulers
        for i in range(self.num_citizens):

            # citizen_initial_prone_to_complain = citizen_initial_prone_to_complain
            if i < self.num_indifferent_citizens:
                # indifferent
                citizen_initial_prone_to_complain = CitizenMemoryInitial.Indifferent.value
            elif i < self.num_indifferent_citizens + self.num_corrupted_citizens:
                citizen_initial_prone_to_complain = CitizenMemoryInitial.Corrupt.value
            else:
                citizen_initial_prone_to_complain = CitizenMemoryInitial.Honest.value

            citizen = Citizen(i,
                              self,
                              cost_accept_mean_std=self.cost_accept_mean_std,
                              prone_to_complain=citizen_initial_prone_to_complain,
                              complain_memory_discount_factor=self.citizen_complain_memory_discount_factor)
            self.schedule.add(citizen)
        # needed for assigning to teams
        self.lookup_corrupt_cops = defaultdict(list)
        for i in range(self.num_cops):
            if i < self.num_indifferent_cops:
                # indifferent
                accepted_bribe_memory_initial = CopMemoryInitial.Indifferent.value
            elif i < self.num_indifferent_cops + self.num_honest_cops:
                accepted_bribe_memory_initial = CopMemoryInitial.Corrupt.value
            else:
                accepted_bribe_memory_initial = CopMemoryInitial.Honest.value

            cop = Cop(i,
                      self,
                      time_left_in_jail=self.initial_time_left_in_jail,
                      accepted_bribe_memory_size=self.memory_size,
                      bribe_amount=self.bribe_amount,
                      moral_commitment_mean_std=self.moral_commitment_mean_std,
                      accepted_bribe_memory_initial=accepted_bribe_memory_initial)
            self.schedule_Cop.add(cop)
            self.lookup_corrupt_cops[
                "corrupt" if accepted_bribe_memory_initial == CopMemoryInitial.Corrupt.value else "other"].append(i)

    def get_citizen(self):
        """
        Get the citizen chosen citizens and give them to the cop.
        :return: citizen
        """
        citizen = random.sample(self.citizens_playing, 1)[0]
        # remove the citizen so they don't get caught twice in the same iteration
        self.citizens_playing.remove(citizen)
        return citizen

    def create_network(self):
        """
        Create network of police officers. They form not intersecting groups of team_size.
        The amount of bribing cops depend on the `initial_indifferent_corruption_honest_rate`
         and how many in each team `corruption_among_teams_spread`.

        NOTE: indifferent and honest cops are treated the same here. First corrupted cops are allocated and then non-corrupt.
        They're allocated randomly. You can set indifferent rate or the honest rate to 0 and then be sure.
        """

        # depending on this check rate of corruption in the team, maybe if none do it totally randomly?
        corrupt_cop_per_team = int(self.num_corrupted_cops / self.number_of_corrupted_teams)
        #  some teams might have to take the additional cops
        surplas_modulo = self.num_corrupted_cops % self.number_of_corrupted_teams

        # Initialise the dictionaries to convert the cop_id to team name, and to convert team name to #cops in jail
        self.id_team = defaultdict(str)
        self.team_jailed = defaultdict(int)
        random.shuffle(self.lookup_corrupt_cops["other"])
        # Corrupted teams first
        for team_number in range(self.number_of_corrupted_teams):
            team_name = "team_" + str(team_number)
            self.team_jailed[team_name] = 0

            number_of_corrupt_cops_in_this_team = corrupt_cop_per_team + (1 if team_number < surplas_modulo else 0)
            # First allocate corrupted cops
            for corrupted_cop in range(number_of_corrupt_cops_in_this_team):
                cop_id = self.lookup_corrupt_cops["corrupt"].pop()
                self.id_team[cop_id] = team_name
            # Allocate not corrupted cops
            for other_cops in range(self.team_size - number_of_corrupt_cops_in_this_team):
                cop_id = self.lookup_corrupt_cops["other"].pop()
                self.id_team[cop_id] = team_name
        # Not Corrupted teams
        for team_number in range(self.number_of_corrupted_teams, self.number_of_teams):
            team_name = "team_" + str(team_number)
            self.team_jailed[team_name] = 0

            # Allocate not corrupted cops
            for other_cops in range(self.team_size):
                # random because some are indifferent and some are honest
                indx = random.randint(0, len(self.lookup_corrupt_cops["other"]) - 1)
                cop_id = self.lookup_corrupt_cops["other"].pop(indx)
                self.id_team[cop_id] = team_name

    def num_active_citizens(self):
        return sum([1 for cit in self.schedule.agents if
                    cit.action is not None])

    def update_network(self):
        """
        Update the current jail rate for each team.
        """
        # Reset the #cops in jail for each team
        for team in self.team_jailed:
            self.team_jailed[team] = 0

        # Update the #cops in jail for each team
        for cop in self.schedule_Cop.agents:
            team = self.id_team[cop.unique_id]
            self.team_jailed[team] += 1 if cop.time_left_in_jail > 0 else 0

    def init_logger(self):
        """
        Logs data in the beginning. Model params and each agent params. Saves it self.log_path at init_params key.
        """
        data_dir = Path("results/")
        data_dir.mkdir(exist_ok=True)
        self.log_path = Path(data_dir, self.experiment_name + '.json')

        with open(self.log_path, 'w') as f:
            # save all init parameters
            log_dict = defaultdict(dict)
            name = 'init_params'
            log_dict = self.get_log_data(name, log_dict)

            json.dump(log_dict, f)

    def log_data(self, step):
        """
        Logs data in each step. Model params and each agent params. Saves it self.log_path at iteration_step key.
        :param step: current iteration
        """

        with open(self.log_path, 'r') as f:
            log_dict = json.load(f)
        name = 'iteration_' + str(step)
        log_dict = self.get_log_data(name, log_dict)

        with open(self.log_path, 'w') as f:
            json.dump(log_dict, f)

    def get_log_data(self, name: str, log_dict: dict) -> dict:
        """
        Collects data from class fields, throws away unnecessary fields or such that are not easily serializable.
        :return: dict with data
        """

        log_dict[name] = deepcopy(vars(self))
        log_dict[name].pop('random', None)
        log_dict[name].pop('running', None)
        log_dict[name].pop('current_id', None)
        log_dict[name].pop('experiment_name', None)
        log_dict[name].pop('log_path', None)
        log_dict[name].pop('schedule', None)
        log_dict[name].pop('schedule_Cop', None)
        log_dict[name].pop('datacollector', None)
        log_dict[name].pop('lookup_corrupt_cops', None)
        log_dict[name].pop('citizens_playing', None)
        log_dict[name].pop('cops_playing', None)
        log_dict[name].pop('logger', None)
        if 'init' not in name:
            # I'm removing those that shouldn't be changed in later steps or it wouldn't change anything if they were
            # rest I left just in case
            log_dict[name].pop('_seed', None)
            log_dict[name].pop('initial_time_left_in_jail', None)
            log_dict[name].pop('cost_accept_mean_std', None)
            log_dict[name].pop('moral_commitment_mean_std', None)
            log_dict[name].pop('fine_amount', None)
            log_dict[name].pop('num_citizens', None)
            log_dict[name].pop('num_cops', None)
            log_dict[name].pop('num_indifferent_cops', None)
            log_dict[name].pop('num_corrupted_cops', None)
            log_dict[name].pop('num_honest_cops', None)
            log_dict[name].pop('num_indifferent_citizens', None)
            log_dict[name].pop('num_corrupted_citizens', None)
            log_dict[name].pop('num_honest_citizens', None)
            log_dict[name].pop('number_of_teams', None)
            log_dict[name].pop('corruption_among_teams_spread', None)
            log_dict[name].pop('number_of_corrupted_teams', None)

        # add agents stats
        log_dict[name]['citizens'] = {}
        log_dict[name]['cops'] = {}
        for cit in self.schedule.agents:
            log_dict[name]['citizens'][cit.unique_id] = cit.log_data()
            if 'init' not in name:
                log_dict[name]['citizens'][cit.unique_id].pop('cost_complain', None)
                log_dict[name]['citizens'][cit.unique_id].pop('rationality', None)
                log_dict[name]['citizens'][cit.unique_id].pop('fine_amount', None)
                log_dict[name]['citizens'][cit.unique_id].pop('penalty_citizen_prosecution', None)
                log_dict[name]['citizens'][cit.unique_id].pop('discount_factor', None)

        for cop in self.schedule_Cop.agents:
            log_dict[name]['cops'][cop.unique_id] = cop.log_data()
            if 'init' not in name:
                log_dict[name]['citizens'][cop.unique_id].pop('jail_cost', None)
                log_dict[name]['citizens'][cop.unique_id].pop('rationality', None)
                log_dict[name]['citizens'][cop.unique_id].pop('jail_cost', None)
                log_dict[name]['citizens'][cop.unique_id].pop('jail_cost', None)

        return log_dict

    def get_server_data_collector(self):
        return DataCollector(
            {"Prison Count": lambda m: sum([1 for cop in self.schedule_Cop.agents if
                                            cop.time_left_in_jail > 0]) / self.schedule_Cop.get_agent_count(),
             "Bribing": lambda m: sum([1 for cop in self.cops_playing if
                                       cop.action == CopActions.bribe]) / sum([1 for cop in self.schedule_Cop.agents if
                                                                               cop.time_left_in_jail == 0]),
             "AcceptComplain": lambda m: self.num_active_citizens() and sum([1 for cit in self.schedule.agents if
                                                                             cit.action == CitizenActions.accept_complain]) / self.num_active_citizens() or 0,
             "Reject_Complain": lambda m: self.num_active_citizens() and sum([1 for cit in self.schedule.agents if
                                                                              cit.action == CitizenActions.reject_complain]) / self.num_active_citizens() or 0,
             "Accept_Silent": lambda m: self.num_active_citizens() and sum([1 for cit in self.schedule.agents if
                                                                            cit.action == CitizenActions.accept_silent]) / self.num_active_citizens() or 0,
             "Reject_Silent": lambda m: self.num_active_citizens() and sum([1 for cit in self.schedule.agents if
                                                                            cit.action == CitizenActions.reject_silent]) / self.num_active_citizens() or 0,
             "Total Complain": lambda m: self.num_active_citizens() and sum([1 for cit in self.schedule.agents if
                                                                             cit.action == CitizenActions.accept_complain or cit.action == CitizenActions.reject_complain
                                                                             ]) / self.num_active_citizens() or 0,
             "Total Accept": lambda m: self.num_active_citizens() and sum([1 for cit in self.schedule.agents if
                                                                           cit.action == CitizenActions.accept_complain or cit.action == CitizenActions.accept_silent
                                                                           ]) / self.num_active_citizens() or 0,
             })
