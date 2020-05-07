import random
import loanrequest
import torch
import pandas as pd
import math
import statistics

OPTIONS_PER_FEATURE = 10 #the number of different actions we want to allow for each of the features
NUMBER_OF_FEATURES = 21

class Action:
    def __init__(self, action_id, action_name, action_value, cost_value, feature_number):
        self.action_id = action_id
        self.action_name = action_name
        self.action_value = action_value
        self.cost = cost_value
        self.feature = feature_number

for i in range(NUMBER_OF_FEATURES):
    for j in range(OPTIONS_PER_FEATURE):
        TempAction = Action(str(i)+str(j),str('multiply feature ')+int(i)+str('by')+float(1+0.01*j) , float(1+0.01*j), j, i)
        actions.append(TempAction)

print(actions)
#actions = [
#    Action(1,'raise credit score by factor 1.1', 1.1, 9, 9),
#    Action(2,'raise credit score by factor 1.2', 1.2, 9, 9)
#]
NUMBER_OF_ACTIONS = len(actions)
MEAN_ACTION_COST = statistics.mean([action.cost for action in actions]) # List Comprehension - create a list of only the costs of all of the actions

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

    ITERS_NUM = 10
    for i in range(ITERS_NUM):

        print('iteration', i)

        # Selection stage
        path_to_leaf = selection(root)
        if len(path_to_leaf) > 1:
            leaf = path_to_leaf[-1]
        else:
            leaf = path_to_leaf[0]

        # Expansion Stage
        next_child = expansion(leaf)

        # Simulation Stage
        total_cost, all_actions, is_successful = simulation(next_child)

        # Back propogation
        path_to_leaf.append(next_child)
        backpropogation(path_to_leaf, is_successful)


def selection(node):

    # If we reach a leaf, move on to expansion
    if len(node.child) == 0:
        return [node]


    # Weight parameters
    c = math.sqrt(2)
    d = math.sqrt(2)
    e = math.sqrt(2)

    max_score = -math.inf
    max_score_idx = -1
    for i, child in enumerate(node.child):
        score = child.num_of_successes / (child.num_of_passes + 1)
        score += c * math.sqrt(math.log(node.num_of_passes + 1) / (child.num_of_passes + 1))
        score -= d * child.total_cost / (MEAN_ACTION_COST * child.depth)
        #score _-= e * something
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
                child.depth = root.depth + 1
                child.total_cost += action.cost
                child.data[child.action.feature] = child.action.action_value * child.data[child.action.feature]
                added_children += 1

    # Pick random child
    return random.choice(leaf.child)


def simulation(node):

    if node.depth == 4:
        return node.total_cost, [node.action], False

    #TODO - change the function's name after we export the loanrequest weights:
    if net.forward(node.data) <= 0.5:
       print("Loan is approved given your current financial status")
       return node.total_cost, [node.action], True

    current_action = random.choice(actions)
    print('Trying action:', current_action.action_name)

    node.child.append(Tree(node.data)) # Add child to the current node
    child = node.child[-1]
    child.action = current_action
    child.depth = root.depth + 1
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
        print(node.num_of_passes)




df = pd.read_csv('dataset\montecarlo_trial.csv')

features_np_array = (df.iloc[0, :-1]).astype(float).to_numpy()
features_tensor = torch.from_numpy(features_np_array).type(torch.FloatTensor)
root = Tree(features_tensor)

net = loanrequest.Net(DROPOUT_RATE=0.1)
net.load_state_dict(torch.load('models\split_33_66_batchsize_500_lr_0.001_dropout_0.1_epoch_1.pkl', map_location='cpu'))

monte_carlo_tree_search(root)