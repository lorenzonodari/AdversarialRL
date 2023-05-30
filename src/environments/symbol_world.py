from .game_objects import *
from .grid_world import GridWorldParams, GridWorld, run_human_agent


class SymbolWorld(GridWorld):

    def __init__(self, params, *, seed=None):
        super().__init__(params, seed=seed)

    def _get_reward_and_gameover(self):
        # returns the reward and whether the game is over
        i,j = self.agent.i, self.agent.j
        if self.key_color.idem_position(i,j):
            return 0, False
        events = self.get_events()
        if self.target_color in events and self._get_room(i,j) in self.target_room:            
            return 1, True # right color and room
        for c in self.colors:
            if c.upper() in events:
                return -1, True # wrong color
        return 0, False

    def get_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = "" 

        # adding the observed events
        i,j = self.agent.i, self.agent.j
        room_agent = self._get_room(i,j)
        ret += str(room_agent)
        for o in self.objects:
            if room_agent == self._get_room(o.i,o.j):
                e_pos = str(o)
                if e_pos != " ":
                    if o.idem_position(i,j):
                        e_pos = e_pos.upper() # the agent is at the color
                    ret += e_pos

        return ret

    def get_all_events(self):
        """
        Returns a string with all the possible events that may occur in the environment
        """
        return "0123abcns.ABC"

    def get_map_classes(self):
        """
        Returns the string with all the classes of objects that are part of this domain
        """
        return "abcns."

    def _get_features_pos_and_dims(self):
        a = self.agent
        room_agent = self._get_room(a.i, a.j)
        
        # adding position of the agent
        dims = [self.max_i, self.max_j]
        pos  = [a.i, a.j]

        # adding the key color
        room_kc  = self._get_room(self.key_color.i,self.key_color.j)
        color_kc = str(self.key_color)
        dims.append(4)
        if room_agent != room_kc: pos.append(0)
        if room_agent == room_kc and color_kc == 'a': pos.append(1)
        if room_agent == room_kc and color_kc == 'b': pos.append(2)
        if room_agent == room_kc and color_kc == 'c': pos.append(3)

        # adding room
        room_kr  = self._get_room(self.key_room.i,self.key_room.j)
        color_kr = str(self.key_room)
        dims.append(4)
        if room_agent != room_kr: pos.append(0)
        if room_agent == room_kr and color_kr == 'n': pos.append(1)
        if room_agent == room_kr and color_kr == 's': pos.append(2)
        if room_agent == room_kr and color_kr == '.': pos.append(2)

        return pos, dims

    def _load_map(self, file_map):
        # loading a map from the set of possible maps
        super()._load_map(file_map)

        # adding problem-specific attributes
        self.objects = []
        self.colors = list("abc")
        self.arrows = list("ns.") # north, south, or any
        for row in self.map:
            for obj in row:
                if str(obj) in "abc?.": self.objects.append(obj)
                if str(obj) in "?": self.key_color = obj
                if str(obj) in ".": self.key_room = obj

        # randomly picking a target color
        self.key_color.label = self._random.choice(self.colors)
        self.key_room.label = self._random.choice(self.arrows)
        self.target_color = str(self.key_color).upper()
        if str(self.key_room) == ".": self.target_room = [0,2]
        if str(self.key_room) == "n": self.target_room = [0]
        if str(self.key_room) == "s": self.target_room = [2]

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
            if room_agent in [0,2]:
                return self._go_to_room(1)
            if room_agent == 1:
                return self._go_to_room(3)
            assert False, "ERRORR"
        if room_agent == 3:
            return self._go_to_room(1)
        if room_agent == 1:
            if 1 <= u <= 3:
                if i > 8: return self._go_to_room(2)
                return self._go_to_room(0)
            if 4 <= u <= 6:
                return self._go_to_room(0)
            if 7 <= u <= 9:
                return self._go_to_room(2)
        if room_agent == 0:
            if 7 <= u <= 9:
                return self._go_to_room(1)
            if i == 4: return Actions.up
            if u % 3 == 1: # a
                if j == 1: return Actions.up
                if (i,j) == (2,3): return Actions.down
                return Actions.left
            if u % 3 == 2: # b
                if j == 2: return Actions.up
                if j == 1: return Actions.right
                if j == 3: return Actions.left
            if u % 3 == 0: # c
                if j == 3: return Actions.up
                if (i,j) == (2,1): return Actions.down
                return Actions.right
            
        if room_agent == 2:
            if 4 <= u <= 6:
                return self._go_to_room(1)
            if i == 12: return Actions.down
            if u % 3 == 1: # a
                if j == 1: return Actions.down
                if (i,j) == (14,3): return Actions.up
                return Actions.left
            if u % 3 == 2: # b
                if j == 2: return Actions.down
                if j == 1: return Actions.right
                if j == 3: return Actions.left
            if u % 3 == 0: # c
                if j == 3: return Actions.down
                if (i,j) == (14,1): return Actions.up
                return Actions.right
        
        assert False, "No action"

    def get_perfect_rm(self):
        # NOTE: This is used for debugging purposes and to compute the expected reward of an optimal policy
        delta_u = {}
        delta_u[(0, '3a.')] = 1
        delta_u[(0, '3b.')] = 2
        delta_u[(0, '3c.')] = 3
        delta_u[(0, '3an')] = 4
        delta_u[(0, '3bn')] = 5
        delta_u[(0, '3cn')] = 6
        delta_u[(0, '3as')] = 7
        delta_u[(0, '3bs')] = 8
        delta_u[(0, '3cs')] = 9
        return delta_u

# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    game_type = "symbolworld"
    file_map = "../../maps/symbol.txt"
    max_time = 5000
    num_total_steps = 1000000
    num_steps = 0

    traces = []
    params = GridWorldParams(game_type, file_map, 0)
    while num_steps < num_total_steps:
        game = SymbolWorld(params)
        reward,steps,trace = run_human_agent(game, max_time)
        num_steps += steps