from .game_objects import *
from .grid_world import GridWorldParams, GridWorld, run_human_agent


class CookieWorld(GridWorld):

    def __init__(self, params, *, seed=None):

        super().__init__(params, seed=seed)

    def _get_reward_and_gameover(self):
        # returns the reward and whether the game is over
        # NOTE: This domain has no game over
        if "C" in self.get_events():
            return 1, False 
        return 0, False

    def get_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = "" 

        # adding the room
        i,j = self.agent.i, self.agent.j
        room_agent = self._get_room(i,j)
        ret += str(room_agent)

        # adding the cookies and button
        for o in self.cookies + [self.button]:
            if room_agent == self._get_room(o.i,o.j):
                e_pos = str(o)
                if e_pos != " ":
                    if o.idem_position(i,j):
                        e_pos = e_pos.upper() # the agent is at the same location than the object
                    ret += e_pos

        return ret

    def get_all_events(self):
        """
        Returns a string with all the possible events that may occur in the environment
        """
        return "0123bcBC"

    def get_map_classes(self):
        """
        Returns the string with all the classes of objects that are part of this domain
        """
        return "bc"

    def _get_features_pos_and_dims(self):
        a = self.agent
        room_agent = self._get_room(a.i, a.j)
        
        # adding position of the agent
        dims = [self.max_i, self.max_j]
        pos  = [a.i, a.j]

        # adding the cookies
        no_cookies = True
        for c in self.cookies:
            if room_agent == self._get_room(c.i,c.j):
                dims.append(2)
                pos.append(int(c.in_map))
                no_cookies = False
        if no_cookies:
            dims.extend([2,2])
            pos.extend([0,0]) 

        return pos, dims

    def _load_map(self, file_map):
        # loading a map from the set of possible maps
        super()._load_map(file_map)

        # adding problem-specific attributes
        self.cookies = []
        for row in self.map:
            for obj in row:
                if str(obj) == 'b': self.button = obj
                if str(obj) == 'c': self.cookies.append(obj)
                
        # initially, there are no cookies
        for c in self.cookies:
            c.in_map = False
        # adding the cookies to the button
        self.button.add_cookies(self.cookies)

    def get_optimal_action(self):
        # NOTE: This is used for debugging purposes and to compute the expected reward of an optimal policy
        """
        R0: north
        R1: hallway
        R2: south
        R3: East
        """
        u = self.u
        i,j = self.agent.i, self.agent.j
        room_agent  = self._get_room(i,j)
        if u == 0:
            if room_agent in [0,2]: return self._go_to_room(1)
            if room_agent == 1: return self._go_to_room(3)
            # press the buttom
            if i < 8: return Actions.down
            if i > 8: return Actions.up
            return Actions.right
        if u == 1:
            # look for a cookie
            if room_agent == 3: 
                return self._go_to_room(1)
            if room_agent == 1: 
                if i > 8: return self._go_to_room(2)
                return self._go_to_room(0)
            assert False, "ERROR2"            
        # States 2 and 3 must go to the hallway from u=3
        if room_agent == 3: 
            return self._go_to_room(1)

        if u == 2:
            # cookie at room 0
            if room_agent == 1: return self._go_to_room(0)
            if room_agent == 2: return self._go_to_room(1)

        if u == 3:
            # cookie at room 2
            if room_agent == 1: return self._go_to_room(2)
            if room_agent == 0: return self._go_to_room(1)

        # go to cookie
        c = [c for c in self.cookies if c.in_map][0]
        if i < c.i: return Actions.down
        if i > c.i: return Actions.up
        if j < c.j: return Actions.right
        if j > c.j: return Actions.left
        
        assert False, "No action"

    def get_perfect_rm(self):
        # NOTE: This is used for debugging purposes and to compute the expected reward of an optimal policy
        delta_u = {}
        delta_u[(0, '3B')] = 1
        delta_u[(1, '0c')] = 2
        delta_u[(1, '2')]  = 2
        delta_u[(1, '2c')] = 3
        delta_u[(1, '0')]  = 3
        delta_u[(2, '3B')] = 1
        delta_u[(2, '0C')] = 4
        delta_u[(3, '3B')] = 1
        delta_u[(3, '2C')] = 4
        return delta_u

# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    file_map = "maps/cookie.txt"
    game_type = "cookieworld"
    max_time = 5000
    num_total_steps = 1000000
    num_steps = 0

    params = GridWorldParams(game_type, file_map, 0)
    while num_steps < num_total_steps:
        game = CookieWorld(params)
        reward,steps,trace = run_human_agent(game, max_time)
        num_steps += steps