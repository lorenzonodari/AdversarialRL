from enum import Enum
import random

"""
The following classes are the types of objects that we are currently supporting 
"""

class Entity:
    def __init__(self,i,j,k=0): #row and column
        self.i = i
        self.j = j
        self.k = k        

    def change_position(self,i,j,k=0):
        self.i = i
        self.j = j
        self.k = k
        
    def idem_position(self,i,j,k=0):
        return self.i==i and self.j==j and self.k==k

    def interact(self, agent, action):
        return True

    def leaving(self):
        pass

class Agent(Entity):
    def __init__(self,actions,i,j,k=0):
        super().__init__(i,j,k=0)
        self.actions = actions
        self.has_box = False
        self.has_key = False

    def get_actions(self):
        return self.actions

    def is_carrying_box(self):
        return self.has_box

    def is_carrying_key(self):
        return self.has_key

    def __str__(self):
        return "A"

class Obstacle(Entity):
    def __init__(self,i,j,k=0,label="X"):
        super().__init__(i,j,k=0)
        self.label = label

    def interact(self, agent, action):
        return False

    def __str__(self):
        return self.label

class Key(Entity):
    def __init__(self,i,j,k=0):
        super().__init__(i,j,k=0)
        self.in_map = True

    def interact(self, agent, action):
        if self.in_map and not agent.has_key:
            agent.has_key = True
            self.in_map = False        
        return True

    def __str__(self):
        if self.in_map:
            return "k"
        return " "

class Buttom(Entity):
    def __init__(self,i,j,k=0):
        super().__init__(i,j,k=0)
        self.in_map = True
        self.is_pressed = False

    def interact(self, agent, action):
        if not self.in_map:
            return True
        if not self.is_pressed and action == Actions.jump:
            self.door.in_map = False
            self.is_pressed = True
        return True

    def add_door(self, door):
        self.door = door

    def __str__(self):
        if not self.in_map:
            return " "
        if self.is_pressed:
            return "p" # button is pressed
        return "u" # buttom up

class CookieButtom(Entity):
    def __init__(self,i,j,k=0):
        super().__init__(i,j,k=0)

    def interact(self, agent, action):
        # removing all the cookies
        for c in self.cookies:
            c.in_map = False
        # adding one cookie at random
        c = random.choice(self.cookies)
        c.in_map = True
        return True

    def add_cookies(self, cookies):
        self.cookies = cookies

    def __str__(self):
        return "b"

class Door(Entity):
    def __init__(self,i,j,k=0):
        super().__init__(i,j,k=0)
        self.in_map = True

    def interact(self, agent, action):
        if self.in_map and agent.has_key:
            agent.has_key = False
            self.in_map = False
        return not self.in_map

    def __str__(self):
        if self.in_map:
            return "z"
        return " "


class Cookie(Entity):
    def __init__(self,i,j,k=0):
        super().__init__(i,j,k=0)
        self.in_map = True

    def leaving(self):
        if self.in_map:
            self.in_map = False

    def __str__(self):
        if self.in_map:
            return "c"
        return " "


class Empty(Entity):
    def __init__(self,i,j,k=0,label=" "):
        super().__init__(i,j,k=0)
        self.label = label

    def __str__(self):
        return self.label


"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left