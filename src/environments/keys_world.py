from .game_objects import *
from .grid_world import GridWorldParams, GridWorld, run_human_agent


class KeysWorld(GridWorld):

    def __init__(self, params, *, seed=None):
        super().__init__(params, seed=seed)

    def _get_reward_and_gameover(self):
        # returns the reward and whether the game is over
        if "G" in self.get_events():
            return 1, True
        return 0, False

    def get_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = "" 

        # adding the id of the current room
        room_agent = self._get_room(self.agent.i, self.agent.j)
        ret += str(room_agent)

        if self.agent.is_carrying_key():
            ret += "*" # means that the agent is carrying the key

        for k in self.keys:
            if k.in_map and room_agent == self._get_room(k.i,k.j):
                ret += "k" # means that the agent is at the same room than a key
        
        for d in self.doors:
            if d.in_map and room_agent == self._get_room(d.i,d.j):
                ret += "d" # means that the agent is at the same room than a door

        if room_agent == self._get_room(self.goal.i,self.goal.j):
            if self.goal.idem_position(self.agent.i, self.agent.j):
                ret += "G"
            else:
                ret += "g"

        return ret

    def get_all_events(self):
        """
        Returns a string with all the possible events that may occur in the environment
        """
        return "0123*gGkd"

    def get_map_classes(self):
        """
        Returns the string with all the classes of objects that are part of this domain
        """
        return "kzg"

    def _get_features_pos_and_dims(self):
        a = self.agent
        room_agent = self._get_room(a.i, a.j)
        
        # adding position of the agent
        dims = [self.max_i, self.max_j]
        pos  = [a.i, a.j]

        # adding whether it has the key
        dims.append(2)
        pos.append(int(a.is_carrying_key()))
        
        # adding whether there are keys in the current room
        no_keys = True
        for k in self.keys:
            if room_agent == self._get_room(k.i,k.j):
                dims.append(2)
                pos.append(int(k.in_map))
                no_keys = False
        if no_keys:
            dims.extend([2,2])
            pos.extend([0,0])            
        
        # adding whether there are doors in the current room
        for d in self.doors:
            dims.append(2)
            pos.append(int(d.in_map and room_agent == self._get_room(d.i,d.j))) 
        
        return pos, dims

    def _load_map(self, file_map):
        # loading a map from the set of possible maps
        super()._load_map(file_map)

        # adding problem-specific attributes
        self.keys = []
        self.doors = []
        for row in self.map:
            for obj in row:
                if str(obj) == 'g': self.goal = obj
                if str(obj) == 'z': self.doors.append(obj) 
                if str(obj) == 'k': self.keys.append(obj)

        # randomly deciding the location of both keys
        key_location = self._random.choice([0,1,2])
        if key_location == 0 or key_location == 2:
            # removing both keys at room "key_location"
            for k in self.keys:
                if self._get_room(k.i,k.j) != key_location:
                    k.in_map = False
        else:
            # randomly removing one key at each location
            k_r0, k_r2 = [],[]
            for k in self.keys:
                if self._get_room(k.i,k.j) == 0: k_r0.append(k)
                if self._get_room(k.i,k.j) == 2: k_r2.append(k)
            k0 = self._random.choice(k_r0)
            k2 = self._random.choice(k_r2)
            k0.in_map = False
            k2.in_map = False


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

        # open a door
        if self.agent.is_carrying_key():
            if room_agent in [0,2]: 
                return self._go_to_room(1)
            if room_agent == 1: 
                return self._go_to_room(3)
            if i < 8: return Actions.down
            if i > 8: return Actions.up
            return Actions.right


        # go from R3 to R1
        if u in [0,1,2,3,4,5]:
            if room_agent == 3: 
                return self._go_to_room(1)

        # go from R1 to R0 or R2
        if u in [0,1]:
            if room_agent == 1: 
                if i > 8: return self._go_to_room(2)
                return self._go_to_room(0)

        # go from (R1 and R2) to R0
        if u in [2,5]:
            if room_agent == 2: 
                return self._go_to_room(1)
            if room_agent == 1: 
                return self._go_to_room(0)

        # go from (R1 and R0) to R2
        if u in [3,4]:
            if room_agent == 0: 
                return self._go_to_room(1)
            if room_agent == 1: 
                return self._go_to_room(2)

        # pick up the closer key
        if u in [1,2,3,4,5]:
            keys = [(abs(k.i-i) + abs(k.j-j) + 0.0001*self._random.random(), k) for k in self.keys if k.in_map]
            keys.sort()
            k = keys[0][1]
            if i < k.i: return Actions.down
            if i > k.i: return Actions.up
            if j < k.j: return Actions.right
            if j > k.j: return Actions.left

        # go to G
        if u == 6:
            if room_agent in [0,2]: 
                return self._go_to_room(1)
            if room_agent == 1: 
                return self._go_to_room(3)
            if j == 12: return Actions.right
            if j == 13: return Actions.up
            if i < 8: return Actions.down
            if i > 8: return Actions.up
            return Actions.right

        return None

    def get_perfect_rm(self):
        # NOTE: This is used for debugging purposes and to compute the expected reward of an optimal policy
        delta_u = {}
        delta_u[(0, '0k')]=1
        delta_u[(0, '2k')]=1
        delta_u[(0, '0kk')]=2
        delta_u[(0, '2')]=2
        delta_u[(0, '2kk')]=3
        delta_u[(0, '0')]=3
        delta_u[(1, '0*')]=4
        delta_u[(1, '2*')]=5
        delta_u[(2, '0*k')]=5
        delta_u[(3, '2*k')]=4
        delta_u[(4, '2*')]=6
        delta_u[(5, '0*')]=6
        return delta_u

# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    game_type = "keysworld"
    file_map = "../../maps/2-keys.txt"
    max_time = 5000
    num_total_steps = 1000000
    num_steps = 0

    traces = []
    params = GridWorldParams(game_type, file_map, 0)
    while num_steps < num_total_steps:
        game = KeysWorld(params)
        reward,steps,trace = run_human_agent(game, max_time)
        num_steps += steps