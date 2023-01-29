import random

import axelrod as axl
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler

from agents import Citizen, Cop

class Corruption(Model):
    def __init__(self,
                 num_citizens=5000,  # constant
                 num_cops=100,  # constant
                 team_size=2,
                 lambda_=0.8,
                 jail_time=2,
                 prob_caught=0.5,
                 memory_size=5,
                 bribe_amount=5,
                 fine_amount=10,  # dont change this in sensitivity analysis
                 cost_complain=4,
                 cost_accept=1,
                 cost_silence=0,
                 prob_success_complain=1,  # remove this
                 complain_reward=1):

        self.team_size = team_size
        self.jail_time = jail_time
        self.prob_caught = prob_caught
        self.memory_size = memory_size
        self.cost_complain = cost_complain

        # Initialise schedulers
        self.schedule_Citizen = BaseScheduler(self)
        self.schedule_Cop = BaseScheduler(self)

        # Add agents to schedulers
        for i in range(num_citizens):
            citizen = Citizen(i,
                              self,
                              lambda_=lambda_,
                              fine_amount=fine_amount,
                              complain_reward=complain_reward,
                              memory_size=memory_size,
                              cost_accept_mean=0.1,
                              cost_accept_std=0.1,
                              cost_silence_mean=0.1,
                              cost_silence_std=0.1,
                              penalty=0.)
            self.schedule_Citizen.add(citizen)

        for i in range(num_cops):
            cop = Cop(i,
                      self,
                      in_jail=0,
                      lambda_=lambda_,
                      memory_size=memory_size,
                      bribe_amount_mean=0.5,
                      bribe_amount_std=0.1,
                      moral_commitment_mean=0.5,
                      moral_commitment_std=0.1,
                      jail_cost=2)
            self.schedule_Cop.add(cop)

        self.cops_playing = [cop for cop in self.schedule_Cop.agents]

        # Data collector to be able to save the data
        self.datacollector = DataCollector(
            {"Prision Count": lambda m: sum([1 for cop in self.schedule_Cop.agents if
                                             cop.in_jail > 0])/self.schedule_Cop.get_agent_count(),
             "Bribing": lambda m: sum([1 for cop in self.cops_playing if
                                       cop.action == "bribe"]) / num_cops,
             "AcceptComplain": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                              cit.action == "accept_complain"]) / num_cops,
             "Reject_Complain": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                               cit.action == "reject_complain"]) / num_cops,
             "Accept_Silent": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                             cit.action == "accept_silent"]) / num_cops,
             "Reject_Silent": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                             cit.action == "reject_silent"]) / num_cops,
             "Total Complain": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                              (cit.action == "accept_complain" or cit.action == "reject_complain")]) / num_cops,
             "Total Accept": lambda m: sum([1 for cit in self.schedule_Citizen.agents if
                                            (cit.action == "accept_complain" or cit.action == "accept_silent")]) / num_cops,
            })

        # Divide the cops over a network of teams
        self.create_network()

    def step(self):
        self.cops_playing = [cop for cop in self.schedule_Cop.agents if cop.in_jail == 0]
        self.citizens_playing = random.sample(self.schedule_Citizen.agents, len(self.cops_playing))

        self.schedule_Citizen.step()
        self.schedule_Cop.step()

        self.datacollector.collect(self)
        self.update_network()

    def get_citizen(self):
        citizen = random.sample(self.citizens_playing, 1)[0]
        self.citizens_playing.remove(citizen)
        return citizen

    def create_network(self):

        # Initialise the dictionaries to convert the cop_id to team name, and to convert team name to #cops in jail
        self.id_team = {}
        self.team_jailed = {}

        # For each cop save their team name and for each team name initialise the #cops in jail
        for team_number, cut in enumerate(range(0, len(self.schedule_Cop.agents), self.team_size)):
            team_name = "team_" + str(team_number)
            for cop in self.schedule_Cop.agents[cut: cut + self.team_size]:
                self.id_team[cop.unique_id] = team_name
            self.team_jailed[team_name] = 0

    def update_network(self):

        # Reset the #cops in jail for each team
        for team in self.team_jailed:
            self.team_jailed[team] = 0

        # Update the #cops in jail for each team
        for cop in self.schedule_Cop.agents:
            team = self.id_team[cop.unique_id]
            self.team_jailed[team] += 1 if cop.in_jail > 0 else 0