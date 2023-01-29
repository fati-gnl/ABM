import random
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler

from agents import Citizen, Cop
from utils import CitizenActions, CopActions


class Corruption(Model):
    def __init__(self,
                 num_citizens=5000,  # constant
                 num_cops=100,  # constant
                 team_size=20,
                 rationality_of_agents=0.8,  # 0 is random totally
                 jail_time=2,
                 prob_of_prosecution=0.5,
                 memory_size=10,
                 fine_amount=1.,  # don't change this in sensitivity analysis
                 cost_complain=0.4,
                 penalty_citizen_prosecution=0.,
                 jail_cost_factor=1.,
                 # jail cost and jail_time should somehow relate to each other I think, but don't know how exactly
                 cost_accept_mean_std=(0.1, 0.1),
                 cost_silence_mean_std=(0.1, 0.1),
                 citizen_initial_prone_to_complain=0.5,
                 citizen_complain_memory_discount_factor=0.5,
                 bribe_amount_mean_std=(0.5, 0.1),
                 moral_commitment_mean_std=(0.5, 0.1),
                 initial_time_left_in_jail=0  # don't think it's worth to change that
                 ):
        super().__init__()

        self.team_size = team_size
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
        self.schedule_Citizen = BaseScheduler(self)
        self.schedule_Cop = BaseScheduler(self)

        # Add agents to schedulers
        for i in range(num_citizens):
            citizen = Citizen(i,
                              self,
                              cost_accept_mean_std=cost_accept_mean_std,
                              cost_silence_mean_std=cost_silence_mean_std,
                              prone_to_complain=citizen_initial_prone_to_complain,
                              complain_memory_discount_factor=citizen_complain_memory_discount_factor)
            self.schedule_Citizen.add(citizen)

        for i in range(num_cops):
            cop = Cop(i,
                      self,
                      time_left_in_jail=initial_time_left_in_jail,
                      accepted_bribe_memory_size=memory_size,
                      bribe_amount_mean_std=bribe_amount_mean_std,
                      moral_commitment_mean_std=moral_commitment_mean_std)
            self.schedule_Cop.add(cop)

        # Data collector to be able to save the data
        self.datacollector = DataCollector(
            {"Prison Count": lambda m: sum([1 for cop in self.schedule_Cop.agents if
                                            cop.time_left_in_jail > 0]) / self.schedule_Cop.get_agent_count(),
             "Bribing": lambda m: sum([1 for cop in self.cops_playing if
                                       cop.action == CopActions.bribe]) / sum([1 for cop in self.schedule_Cop.agents if
                                                                               cop.time_left_in_jail == 0]),
             "AcceptComplain": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                              cit.action == CitizenActions.accept_complain]) / self.num_active_citizens(),
             "Reject_Complain": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                               cit.action == CitizenActions.reject_complain]) / self.num_active_citizens(),
             "Accept_Silent": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                             cit.action == CitizenActions.accept_silent]) / self.num_active_citizens(),
             "Reject_Silent": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                             cit.action == CitizenActions.reject_silent]) / self.num_active_citizens(),
             "Total Complain": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                              cit.action == CitizenActions.accept_complain or cit.action == CitizenActions.reject_complain
                                              ]) / self.num_active_citizens(),
             "Total Accept": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                            cit.action == CitizenActions.accept_complain or cit.action == CitizenActions.accept_silent
                                            ]) / self.num_active_citizens(),
             })

        # Divide the cops over a network of teams
        self.create_network()

    def step(self):
        self.cops_playing = [cop for cop in self.schedule_Cop.agents if cop.time_left_in_jail == 0]
        self.citizens_playing = random.sample(self.schedule_Citizen.agents, len(self.cops_playing))

        self.schedule_Citizen.step()
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
        # Initialise the dictionaries to convert the cop_id to team name, and to convert team name to #cops in jail
        self.id_team = {}
        self.team_jailed = {}

        # For each cop save their team name and for each team name initialise the #cops in jail
        for team_number, cut in enumerate(range(0, len(self.schedule_Cop.agents), self.team_size)):
            team_name = "team_" + str(team_number)
            for cop in self.schedule_Cop.agents[cut: cut + self.team_size]:
                self.id_team[cop.unique_id] = team_name
            self.team_jailed[team_name] = 0

    def num_active_citizens(self):
        return sum([1 for cit in self.schedule_Citizen.agents if
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
