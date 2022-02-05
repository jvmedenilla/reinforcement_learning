import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        
        From utils.py:
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3
        
        python3 mp7.py --snake_head_x 40 --snake_head_y 80 --food_x 120 --food_y 120 --Ne 40 --C 40 --gamma 0.7
        python compare_tables.py --test 1 --checkpoint checkpoint.npy --checkpoint-n checkpoint_N.npy 
        
        '''
    

        s_prime = self.generate_state(environment)  # next state from taking action a

        # calculate reward from taking action a:
        if dead == True:  # that means the snake died upon taking action a
            R = -1
        if points > self.points:  # that means the the snake ate a pellet upon taking action a
            R = 1 
            self.points = points      # update self.points to match the parameter points
        if points == self.points and dead == False: # did not die nor get pellet
            R = -0.1
        
        maxQ = max(self.Q[s_prime])
                    
        if self._train == True:
            if self.s != None and self.a != None:
                self.N[self.s][self.a] += 1   # update the N table
                alpha = self.C / (self.C + self.N[self.s][self.a])
                # updating the Q-table:
                self.Q[self.s][self.a] =  self.Q[self.s][self.a] + alpha*(R + self.gamma*(maxQ) - self.Q[self.s][self.a])
        
            # calculate a_prime:
            a_array = np.zeros((4))
            one_counter = 0
            for action in self.actions:
                if self.N[s_prime][action] < self.Ne:
                    a_array[action] = 1
                    one_counter += 1
                else:
                    a_array[action] = self.Q[s_prime][action] 

            if one_counter > 1:
                for nums in range(len(a_array)):
                    if a_array[nums] == 1:  # choose the last action that has 1
                        a_prime = nums
            else:
                a_max = max(a_array)
                a_list = list(a_array)
                if a_list.count(a_max) > 1:   # to break ties in max Q values
                    for nums in range(len(a_array)):
                        if a_array[nums] == a_max:
                            a_prime = nums
                else:
                    a_prime = np.argmax(a_array)
                
            
            if dead == False:     
                self.s = s_prime
                self.a = a_prime
                
            if dead == True:
                self.reset()
                self.a = None
        else:       # if in testing mode, just get the argmax of the Qvalues
            #a_max1 = max(self.Q[s_prime])
            #a_list1 = list(self.Q[s_prime])
            #if a_list1.count(a_max1) > 1:   # to break ties in max Q values
            #    for nums in range(len(a_list1)):
            #        if self.Q[s_prime][nums] == a_max1:
            #            a_prime = nums
            #else:           # if there's only 1 max Q value, then return argmax
            #    a_prime = np.argmax(self.Q[s_prime])
            a_prime = np.argmax(self.Q[s_prime])
            
        return a_prime
    def generate_state(self, environment):
        '''
        convert the given argument into a state to be used in the MDP
        environment = [snake_head_x, snake_head_y, snake_body, food_x, food_y]
        '''
        
        # assign variables to environment parts
        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]
        
        
        # populate food_dir_x
        if snake_head_x == food_x:
            food_dir_x = 0     # same x-axis
        if snake_head_x > food_x:
            food_dir_x = 1     # food is left of head
        if snake_head_x < food_x:
            food_dir_x = 2     # food is right of head
        # populate food_dir_y    
        if snake_head_y == food_y:
            food_dir_y = 0     # same y-axis
        if snake_head_y > food_y:
            food_dir_y = 1     # food is top of head
        if snake_head_y < food_y:
            food_dir_y = 2     # food is below head
        # populate adjoining_wall_x
        adjoining_wall_x = 0
        if snake_head_x == utils.WALL_SIZE:
            adjoining_wall_x = 1   # Wall is on head's left
        if snake_head_x == utils.DISPLAY_SIZE - 2*utils.WALL_SIZE:
            adjoining_wall_x = 2   # Wall is on head's right
        #if utils.WALL_SIZE < snake_head_x < utils.DISPLAY_SIZE - 2*utils.WALL_SIZE:            
        #else:  
        #    adjoining_wall_x = 0   # Head is not adjacent to any walls
        # populate adjoining_wall_y
        adjoining_wall_y = 0
        if snake_head_y == utils.WALL_SIZE:
            adjoining_wall_y = 1   # Wall is on head's top
        if snake_head_y == utils.DISPLAY_SIZE - 2*utils.WALL_SIZE:
            adjoining_wall_y = 2   # Wall is on head's bottom
        #if utils.WALL_SIZE < snake_head_y < utils.DISPLAY_SIZE - 2*utils.WALL_SIZE:  
        #else:            
        #    adjoining_wall_y = 0   # Head is not adjacent to any walls

        # check for adjoining_top
        if (snake_head_x, snake_head_y - utils.GRID_SIZE) in snake_body:
            adjoining_body_top = 1
        else: 
            adjoining_body_top = 0
        # check for adjoining_bottom
        if (snake_head_x, snake_head_y + utils.GRID_SIZE) in snake_body:
            adjoining_body_bottom = 1
        else: 
            adjoining_body_bottom = 0
        # check for adjoining_left
        if (snake_head_x - utils.GRID_SIZE, snake_head_y) in snake_body:
            adjoining_body_left = 1
        else: 
            adjoining_body_left = 0
        # check for adjoining_right
        if (snake_head_x + utils.GRID_SIZE, snake_head_y) in snake_body:
            adjoining_body_right = 1
        else: 
            adjoining_body_right = 0
            
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)