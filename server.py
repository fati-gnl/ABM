import mesa
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from model import *

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

chart2 = ChartModule([{"Label": "Prison Count",
                       "Color": "red"}],
                     data_collector_name='datacollector')

chart_complain = ChartModule([{"Label": "Total Complain",
                               "Color": "red"}],
                             data_collector_name='datacollector')

chart_accept = ChartModule([{"Label": "Total Accept",
                             "Color": "green"}],
                           data_collector_name='datacollector')

model_params = {
    # The following line is an example to showcase StaticText.
    "title": mesa.visualization.StaticText("Model Parameters"),
    "team_size": mesa.visualization.Slider(
        "Cop Network Size", 10, 1, 100, 1,
        description="The number of cops in a network.",
    ),
    "jail_time": mesa.visualization.Slider(
        "Jail Time", 1, 0, 5, 1
    ),
    "cost_complain": mesa.visualization.Slider(
        "Cost of Complaining", 0.50, 0., 1.0, 0.01
    ),
}


# Create the server, and pass the grid and the graph
server = ModularServer(Corruption,
                       [chart2, chart4, chart5, chart_complain, chart_accept],
                       "Corruption Model", model_params, {})


server.port = 8526

server.launch()
