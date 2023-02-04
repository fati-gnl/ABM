import math
import random
from collections import defaultdict

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler

from agents import Citizen, Cop
from utils import CitizenActions, CopActions, CitizenMemoryInitial, CopMemoryInitial


class Corruption(Model):
    def __init__(self,
                 num_citizens=50000,
                 num_cops=1000,
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
                 initial_indifferent_corruption_honest_rate=(0.8, 0.2, 0),
                 corruption_among_teams_spread=1.0
                 # rate of teams that should be getting the corrupted cops. 1 - all teams have the same amount(+-1 cop ofc)
                 ):

        super().__init__()

        self.num_citizens = num_citizens
        self.num_cops = num_cops

        self.num_indifferent_cops = int(initial_indifferent_corruption_honest_rate[0] * num_cops)
        self.num_corrupted_cops = int(initial_indifferent_corruption_honest_rate[1] * num_cops)
        self.num_honest_cops = int(initial_indifferent_corruption_honest_rate[2] * num_cops)
        self.num_honest_cops += self.num_cops - (self.num_corrupted_cops + self.num_honest_cops + self.num_honest_cops)

        self.num_indifferent_citizens = int(initial_indifferent_corruption_honest_rate[0] * num_citizens)
        self.num_corrupted_citizens = int(initial_indifferent_corruption_honest_rate[1] * num_citizens)
        self.num_honest_citizens = int(initial_indifferent_corruption_honest_rate[2] * num_citizens)
        self.num_honest_citizens += self.num_citizens - (
                self.num_corrupted_citizens + self.num_honest_citizens + self.num_honest_citizens)

        self.team_size = team_size
        self.number_of_teams = math.ceil(self.num_cops / self.team_size)

        # how many iterations cop is inactive
        self.jail_time = jail_time
        # actual cost that cop takes into consideration in the utility function
        self.jail_cost = jail_cost_factor * jail_time

        self.prob_of_prosecution = prob_of_prosecution
        self.memory_size = memory_size
        self.cost_complain = cost_complain
        self.penalty_citizen_prosecution = penalty_citizen_prosecution
        self.fine_amount = fine_amount
        self.rationality_of_agents = rationality_of_agents

        # Initialise schedulers
        self.schedule = BaseScheduler(self)
        self.schedule_Cop = BaseScheduler(self)

        assert sum(
            [rate for rate in initial_indifferent_corruption_honest_rate]) == 1.0, "Distribution should sum up to 1."

        # Checking if the value isn't to small, if it is set it to minimum
        self.number_of_corrupted_teams = math.ceil(max(corruption_among_teams_spread * self.number_of_teams,
                                                       initial_indifferent_corruption_honest_rate[
                                                           1] * num_cops / self.team_size))

        # Add agents to schedulers
        for i in range(num_citizens):

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
                              cost_accept_mean_std=cost_accept_mean_std,
                              prone_to_complain=citizen_initial_prone_to_complain,
                              complain_memory_discount_factor=citizen_complain_memory_discount_factor)
            self.schedule.add(citizen)
        self.lookup_corrupt_cops = defaultdict(list)
        for i in range(num_cops):

            if i < self.num_indifferent_cops:
                # indifferent
                accepted_bribe_memory_initial = CopMemoryInitial.Indifferent.value

            elif i < self.num_indifferent_cops + self.num_honest_cops:
                accepted_bribe_memory_initial = CopMemoryInitial.Corrupt.value

            else:
                accepted_bribe_memory_initial = CopMemoryInitial.Honest.value

            cop = Cop(i,
                      self,
                      time_left_in_jail=initial_time_left_in_jail,
                      accepted_bribe_memory_size=memory_size,
                      bribe_amount=bribe_amount,
                      moral_commitment_mean_std=moral_commitment_mean_std,
                      accepted_bribe_memory_initial=accepted_bribe_memory_initial)
            self.schedule_Cop.add(cop)
            self.lookup_corrupt_cops[
                "corrupt" if accepted_bribe_memory_initial == CopMemoryInitial.Corrupt.value else "other"].append(i)

        # Data collector to be able to save the data
        self.datacollector = DataCollector(
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

        # Divide the cops over a network of teams
        self.create_network()

    def step(self):
        self.cops_playing = [cop for cop in self.schedule_Cop.agents if cop.time_left_in_jail == 0]
        self.citizens_playing = random.sample(self.schedule.agents, len(self.cops_playing))

        self.schedule.step()
        self.schedule_Cop.step()

        self.datacollector.collect(self)
        self.update_network()

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
        Create network of police officers. They form not intersecting groups of team_size
        """

        # depending on this check rate of corruption in the team, maybe if none do it totally randomly?
        corrupt_cop_per_team = int(self.num_corrupted_cops / self.number_of_corrupted_teams)
        #  some teams might have to take the additional cops
        surplas_modulo = self.num_corrupted_cops % self.number_of_corrupted_teams

        # Initialise the dictionaries to convert the cop_id to team name, and to convert team name to #cops in jail
        self.id_team = defaultdict(str)
        self.team_jailed = defaultdict(int)
        for team_number in range(self.number_of_corrupted_teams):
            team_name = "team_" + str(team_number)
            self.team_jailed[team_name] = 0

            number_of_corrupt_cops_in_this_team = corrupt_cop_per_team + (1 if team_number < surplas_modulo else 0)
            # First allocate corrupted cops
            for corrupted_cop in range(number_of_corrupt_cops_in_this_team):
                cop_id = self.lookup_corrupt_cops["corrupt"].pop(0)
                self.id_team[cop_id] = team_name
            # Allocate not corrupted cops
            for other_cops in range(self.team_size - number_of_corrupt_cops_in_this_team):
                # random because some are indifferent and some are honest
                cop_id = self.lookup_corrupt_cops["other"].pop(random.randint(0, len(self.lookup_corrupt_cops["other"])))
                self.id_team[cop_id] = team_name

        for team_number in range(self.number_of_corrupted_teams, self.number_of_teams):
            team_name = "team_" + str(team_number)
            self.team_jailed[team_name] = 0

            # Allocate not corrupted cops
            for other_cops in range(self.team_size):
                # random because some are indifferent and some are honest
                indx = random.randint(0, len(self.lookup_corrupt_cops["other"])-1)
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
