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
chart = ChartModule([{"Label": "Bribing",
                      "Color": "green"}, {"Label": "NoBribing",
                      "Color": "yellow"}],
                    data_collector_name='datacollector')

chart2 = ChartModule([{"Label": "ComplainRate",
                      "Color": "blue"},{"Label": "NoComplainRate",
                      "Color": "red"}],
                    data_collector_name='datacollector')

chart3 = ChartModule([{"Label": "Citizens",
                      "Color": "blue"},{"Label": "Cops",
                      "Color": "red"}],
                    data_collector_name='datacollector')
# chart = ChartModule([{"Label": "Citizens",
#                       "Color": "green"},
#                       {"Label": "Cops",
#                       "Color": "blue"}],
#                     data_collector_name='datacollector')

# Create the server, and pass the grid and the graph
server = ModularServer(CopCitizen,
                       [chart, chart2,chart3],
                       "CopCitizen Model",
                       {})

server.port = 8526

server.launch()
