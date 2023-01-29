from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from model import *
from agents import Cop, Citizen


# Create a Potrayal
def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Color": "blue" if type(agent) is Cop else "green",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}
    return portrayal


# Create a dynamic linegraph
chart4 = ChartModule([{"Label": "Bribing",
                       "Color": "green"}],
                     data_collector_name='datacollector')

chart5 = ChartModule([{"Label": "AcceptComplain",
                       "Color": "blue"},
                      {"Label": "Reject_Complain",
                       "Color": "red"},
                      {"Label": "Accept_Silent",
                       "Color": "green"},
                      {"Label": "Reject_Silent",
                       "Color": "yellow"}],
                     data_collector_name='datacollector')

# chart3 = ChartModule([{"Label": "Citizens",
# "Color": "blue"},{"Label": "Cops",
# "Color": "red"}],
# data_collector_name='datacollector')
chart = ChartModule([{"Label": "Citizens",
                      "Color": "green"},
                     {"Label": "Cops",
                      "Color": "blue"}],
                    data_collector_name='datacollector')

chart2 = ChartModule([{"Label": "Prision Count",
                       "Color": "red"}],
                     data_collector_name='datacollector')

chart_complain = ChartModule([{"Label": "Total Complain",
                               "Color": "red"}],
                             data_collector_name='datacollector')

chart_accept = ChartModule([{"Label": "Total Accept",
                             "Color": "green"}],
                           data_collector_name='datacollector')

# Create the server, and pass the grid and the graph
server = ModularServer(Corruption,
                       [chart2, chart4, chart5, chart_complain, chart_accept],
                       "CopCitizen Model",
                       {})

server.port = 8526

server.launch()
