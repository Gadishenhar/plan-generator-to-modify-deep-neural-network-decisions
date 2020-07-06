import random
import loanrequest
import torch
import pandas as pd
import math
import statistics
import copy
import numpy as np
from datetime import datetime

OPTIONS_PER_FEATURE = 9 #the number of different actions we want to allow for each of the features
NUMBER_OF_FEATURES = 21 #Number of given features
MAX_DEPTH = 3 #The maximal depth of our tree
THRESHOLD = 0.8

class Action:
    def __init__(self, action_id, action_name, action_value, cost_value, feature_number):
        self.action_id = action_id
        self.action_name = action_name
        self.action_value = action_value
        self.cost = cost_value
        self.feature = feature_number

class Tree (object):
    """
    Each node consists of a "grade" on the loan test, which is the ouput of the aquisition net.
    Each edge is a financial action that can be done, and will affect the grade.

    Variables"
   * np.array features - array of the 21 features (TODO might change this to save memory)
   * int network_val - result for running the network on the current features
   * Tree[] children - list of all children

   * Action last_action_taken
   * int total_rollouts
   * int succ_rollouts


    """

    def __init__(self, data):
        self.left = None
        self.child = []
        self.data = data
        self.total_cost = 0
        self.action = None
        self.depth = 0
        self.num_of_successes = 0
        self.num_of_passes=0
    def createChildren(self,amount):
        for i in range (0,amount):
            self.child.append(Tree())
    def setChildrenValues(self,list):
        for i in range(0,len(list)):
            self.data.append(list[i])

def monte_carlo_tree_search(root):

    ITERS_NUM = 20
    for i in range(ITERS_NUM):

        #print('iteration', i)

        # Selection stage
        path_to_leaf = selection(root)
        if len(path_to_leaf) > 1:
            leaf = path_to_leaf[-1]
        else:
            leaf = path_to_leaf[0]

        # Expansion Stage
        if leaf.depth != MAX_DEPTH:
            next_child = expansion(leaf)

        else:
            next_child = leaf

        next_child_sim = copy.copy(next_child)

        # Simulation Stage
        total_cost, all_actions, is_successful = simulation(next_child_sim)

        # Back propogation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        path_to_leaf.append(next_child)
        backpropogation(path_to_leaf, is_successful)

    #Choose the best route of all and propose it to the user:
    proposed_actions = []
    proposed_actions = best_route(root)

    proposed_actions = proposed_actions[:-1]

    if proposed_actions==[]:
        print("Nothing will help you. You're doomed.")
        return

    print("In order to make your mortgage application approved, do the following actions:", str(proposed_actions))

def selection(node):

    # If we reach a leaf, move on to expansion
    if len(node.child) == 0:
        #print('The leaf depth is', node.depth)
        return [node]


    # Weight parameters
    A = 1
    B = math.sqrt(2)
    C = math.sqrt(2)
    D = math.sqrt(2)

    max_score = -math.inf
    max_score_idx = -1
    for i, child in enumerate(node.child):
        score = A * child.num_of_successes / (child.num_of_passes + 1)
        score += B * math.sqrt(math.log(node.num_of_passes + 1) / (child.num_of_passes + 1))
        score -= C * child.total_cost / (MEAN_ACTION_COST * child.depth)
        score -= D * child.depth
        if score > max_score:
            max_score = score
            max_score_idx = i

    result = [node]
    result.extend(selection(node.child[max_score_idx]))
    return result


def expansion(leaf):

    # TODO Only adding 2 children means this node will never have more than two children, we miss possible routes.
    if not len(leaf.child) == NUMBER_OF_ACTIONS:
        # Create N new children
        NUM_OF_CHILDREN = 2
        added_children = 0
        while added_children < NUM_OF_CHILDREN:

            action = random.choice(actions)

            does_exist = False
            for c in leaf.child:
                if c.action.action_id == action.action_id:
                    does_exist = True

            if not does_exist:
                leaf.child.append(Tree(leaf.data))  # Add child to the current node
                child = leaf.child[-1]
                child.action = action
                child.depth = leaf.depth + 1
                child.total_cost += action.cost
                child.data[child.action.feature] = child.action.action_value * child.data[child.action.feature]
                added_children += 1

    # Pick random child
    return random.choice(leaf.child)


def simulation(node):

    if node.depth == MAX_DEPTH:
        #print('Too deep')
        return node.total_cost, [node.action], False

    #TODO - change the function's name after we export the loanrequest weights:

    net_out = float(net.forward((node.data)))

    print(net_out)
    if net_out <= THRESHOLD:
       print("Successful path!")
       return node.total_cost, [node.action], True

    current_action = random.choice(actions)
    #print('Trying action:', current_action.action_name)

    node.child.append(Tree(node.data)) # Add child to the current node
    child = node.child[-1]
    child.action = current_action
    child.depth = node.depth + 1
    child.data[child.action.feature] = child.action.action_value * child.data[child.action.feature]

    total_cost, all_actions, is_successful = simulation(child) #Call this child's child, and save the cost, the list of actions and whether the rollout was successful
    all_actions.append(current_action)
    child.total_cost = total_cost #Summing the cost of this child and its subtrees
    return total_cost + current_action.cost, all_actions, is_successful

def backpropogation(nodes, is_successful):
    if nodes == None:
        return
    for node in nodes:
        node.num_of_passes += 1
        node.num_of_successes += int(is_successful)
       #print(node.num_of_passes)


def best_route(node):

    #print('Searching for best route in depth', node.depth)

    #Recursion stop condition:
    temp = []
    if node.child==[]: #if it's a successful leaf: meaning there are no children
        temp.append(node.action.action_name)
        return temp

    list_of_actions = []
    A = 1
    C = D = math.sqrt(2)
    max_score = -math.inf
    max_score_idx = 1.5
    for i, child in enumerate(node.child):
        if child.num_of_successes==0:
            continue
        score = A * child.num_of_successes / (child.num_of_passes + 1)
        score -= C * child.total_cost / (MEAN_ACTION_COST * child.depth)
        score -= D * child.depth
        if score > max_score:
            max_score = score
            max_score_idx = i

    if max_score_idx==1.5:
        return list_of_actions

    list_of_actions.extend(best_route(node.child[max_score_idx]))
    list_of_actions.append(node.action.action_name)
    return list_of_actions

def generate_actions (feature,values,curr_value, is_discrete):
    actions = []
    for i in values:
        if is_discrete:
            action_value = i / curr_value.iloc[0,feature]
            if i == curr_value.iloc[0,feature]:
                continue
        else:
            action_value = i
        curr_cost = (abs(action_value * curr_value.iloc[0,feature] - stats_mean.iloc[feature+1])) / stats_std.iloc[feature+1]
        TempAction = Action(str(feature) + str(i),
                            str('multiply feature ') + str(feature) + str(' by ') + str(action_value),
                            action_value, curr_cost, feature)
        actions.append(TempAction)
    return actions

df = pd.read_csv('dataset\montecarlo_trial.csv')

#Load the statistics about the data
stats = pd.read_csv('dataset\statistics.csv')
stats_mean = stats.iloc[1]
stats_std = stats.iloc[2]

"""
actions = [] #Initializing an empty list which will contain all the actions
for i in range(1,NUMBER_OF_FEATURES): #i represent the feature which we shift
    if (i==2):
        if (df.iloc[0, i]==0):
            for bank in range(1, 17):
                action_value = bank
                ##TODO - consider the meaning of cost in this case
                curr_cost = (abs(action_value * df.iloc[0, i] - stats_mean.iloc[i])) / stats_std.iloc[i]
                TempAction = Action(str(i) + str(bank),
                                    str('multiply feature ') + str(i) + str(' by ') + str(action_value),
                                    action_value, curr_cost, i)
                actions.append(TempAction)
            continue
        for bank in range(1,17):
            action_value = bank/df.iloc[0, i]
            ##TODO - consider the meaning of cost in this case
            curr_cost = (abs(action_value * df.iloc[0, i] - stats_mean.iloc[i])) / stats_std.iloc[i]
            TempAction = Action(str(i) + str(bank), str('multiply feature ') + str(i) + str(' by ') + str(action_value),
                                action_value, curr_cost, i)
            actions.append(TempAction)
        continue
    if (i==8):
        for Borr in range(1,4):
            action_value = 1 + Borr
            curr_cost = (abs(action_value * df.iloc[0, i] - stats_mean.iloc[i])) / stats_std.iloc[i]
            TempAction = Action(str(i) + str(Borr), str('multiply feature ') + str(i) + str(' by ') + str(action_value),
                                action_value, curr_cost, i)
            actions.append(TempAction)
        continue
    for j in range(1, OPTIONS_PER_FEATURE): #j represent the percentage in which is shift the feature
        action_value=1+0.01*j
        curr_cost = ( abs(action_value*df.iloc[0,i]-stats_mean.iloc[i]) ) / stats_std.iloc[i]
        TempAction = Action(str(i)+str(j),str('multiply feature ')+str(i)+str(' by ')+str(action_value), action_value, curr_cost, i)
        actions.append(TempAction)
"""

actions = [] ##Initializing an empty list of actions
#Generating actions for each feature:
actions.extend(generate_actions(0,[1,2,3],df,True))  # Origination channel
actions.extend(generate_actions(1,list(range(1,97)),df,True))  # Seller name
actions.extend(generate_actions(4,list(np.arange(999,500,-1)/1000),df,False)) # UPB - Decrease by up to 50%
actions.extend(generate_actions(5,list(np.arange(999,500,-1)/1000),df,False)) # LTV - Decrease by up to 50%
actions.extend(generate_actions(6,list(np.arange(999,500,-1)/1000),df,False))  # CLTV
actions.extend(generate_actions(7,[1,2,3],df,True)) ##Number of borrowers
actions.extend(generate_actions(8,list(np.arange(999,500,-1)/1000),df,False))  # Debt to income
actions.extend(generate_actions(9,list(np.arange(1001,1500)/1000),df,False))  # Credit Score
actions.extend(generate_actions(10,[1,2],df,True)) #First time home buyer
actions.extend(generate_actions(11,list(range(1,4)),df,True)) #LOAN PURPOSE
#actions.extend(generate_actions(0,list(range(1,6)),df,True)) #Property type
actions.extend(generate_actions(12,list(range(1,5)),df,True)) # Number of units
actions.extend(generate_actions(13,list(range(1,4)),df,True)) # Occupancy Type
actions.extend(generate_actions(16,list(np.arange(1001,1500)/1000),df,False))  # PRIMARY MORTGAGE INSURANCE PERCENT
actions.extend(generate_actions(18,list(np.arange(1001,1500)/1000),df,False))  # CoBorrower Credit Score
actions.extend(generate_actions(19,list(range(1,4)),df,True)) #MORTGAGE INSURANCE TYPE
actions.extend(generate_actions(20,list(range(1,3)),df,True)) #RELOCATION MORTGAGE INDICATOR

NUMBER_OF_ACTIONS = len(actions)
MEAN_ACTION_COST = statistics.mean([action.cost for action in actions]) # List Comprehension - create a list of only the costs of all of the actions

features_np_array = (df.iloc[0, :-1]).astype(float).to_numpy()
features_tensor = torch.from_numpy(features_np_array).type(torch.FloatTensor)
root = Tree(features_tensor)
root.action = Action(action_id=0, action_name="current_state", action_value=0, cost_value=0, feature_number=0) #We create a fictive action for the root, just to make sure the algorithm runs well. We will delete this from the proposed list.

net = loanrequest.Net(DROPOUT_RATE=0.1)
net.load_state_dict(torch.load('models\split_33_66_batchsize_500_lr_0.001_dropout_0.1_epoch_1.pkl', map_location='cpu'))

monte_carlo_tree_search(root)
