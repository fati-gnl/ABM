from mesa import Agent
import random

class RandomWalker(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos

    def random_move(self):
        '''
        This method should get the neighbouring cells (Moore's neighbourhood), select one, and move the agent to this cell.
        Input:
        Output: Moves the agent to a new neighbouring position
        '''
        # Get the Moore neighborhood cells (including diagonals)
        list_neighbours = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)

        if len(list_neighbours) > 0:
            # Select one random neighbour
            new_pos = self.random.choice(list_neighbours)
            # Move the agent to that cell
            self.model.grid.move_agent(self, new_pos)

class Citizen(RandomWalker):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

    def step(self):
        '''
        This method should move the Citizen using the `random_move()` method,
        then conditionally "".
        Input:
        Output:
        '''
        self.random_move()


class Cop(RandomWalker):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

    def step(self):
        '''
        This method should conditially check if there is a citizen that has committed an offence next to them
        Input:
        Output:
        '''
        self.random_move()