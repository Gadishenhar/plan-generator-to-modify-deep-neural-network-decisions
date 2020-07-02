import random
import loanrequest
import torch
import pandas as pd
import math
import statistics
import copy
import numpy as np
import preprocessor
from visualize_tree import visualize_tree

OPTIONS_PER_FEATURE = 9 #the number of different actions we want to allow for each of the features
NUMBER_OF_FEATURES = 21 #Number of given features
MAX_DEPTH = 3 #The maximal depth of our tree
THRESHOLD = 0.5

class Action:
    def __init__(self, action_id, action_name, action_value, cost_value, feature_number):
        self.action_id = action_id
        self.action_name = action_name
        self.action_value = action_value
        self.cost = cost_value
        self.feature = feature_number

class Tree (object):

    def __init__(self, data):
        self.child = []
        self.data = data.clone().detach()
        self.total_cost = 0
        self.action = None
        self.depth = 0
        self.num_of_successes = 0
        self.num_of_passes=0

def monte_carlo_tree_search(root):

    ITERS_NUM = 5
    for i in range(ITERS_NUM):

        print('Iteration', i)
        input()

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
        print('After Expansion, the tree is')
        visualize_tree(root)
        print('\n\n\n\n\n\n\n\n\n\n\n\n')
        input()

        next_child_sim = copy.copy(next_child)

        # Simulation Stage
        total_cost, all_actions, is_successful = simulation(next_child_sim)

        print('After Simulation, the tree is')
        visualize_tree(root)
        print('\n\n\n\n\n\n\n\n\n\n\n\n')
        input()

        # Back propogation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        path_to_leaf.append(next_child)
        backpropogation(path_to_leaf, is_successful)

        print('After back propogation, the tree is')
        visualize_tree(root)
        print('\n\n\n\n\n\n\n\n\n\n\n\n')
        input()

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

    if not len(leaf.child) == NUMBER_OF_ACTIONS:
        # Create N new children
        # TODO Change this back to 20
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
        return node.total_cost, [node.action], False

    #TODO - change the function's name after we export the loanrequest weights:

    # Tensor to dataframe
    df_data = pd.DataFrame(data=node.data.numpy().copy()).T
    # Normalize
    norm_data = preprocessor.norm_features(df_data, stats)
    # Dataframe to tensor
    norm_data = torch.tensor(norm_data.values).float()
    net_out = float(net.forward(norm_data))

    if net_out <= THRESHOLD:
       return node.total_cost, [node.action], True

    current_action = random.choice(actions)

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


def best_route(node):

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


# For a given user, we want deterministic results
random.seed(0)

#Load the statistics about the data
stats = pd.read_csv('dataset\statistics.csv')
stats_mean = stats.iloc[1]
stats_std = stats.iloc[2]

# Load request data and tokenize
df = pd.read_csv('dataset\montecarlo_trial.csv', names=preprocessor.COL_NAMES)
df = preprocessor.prep_columns(df)



#Generating actions for each feature:
actions = [] ##Initializing an empty list of actions
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
