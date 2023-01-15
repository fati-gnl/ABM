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

# Create a grid of 20 by 20 cells, and display it as 500 by 500 pixels
grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)

# Create a dynamic linegraph
chart = ChartModule([{"Label": "Citizen",
                      "Color": "green"},
                      {"Label": "Cop",
                      "Color": "blue"}],
                    data_collector_name='datacollector')

# Create the server, and pass the grid and the graph
server = ModularServer(CopCitizen,
                       [grid, chart],
                       "CopCitizen Model",
                       {})

server.port = 8526

server.launch()