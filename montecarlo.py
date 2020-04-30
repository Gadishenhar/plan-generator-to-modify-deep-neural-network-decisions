import random
import loanrequest
import torch
import pandas as pd
import xlrd

class Action:
    def __init__(self, action_id, action_name, action_value, cost_value, feature_number):
        self.action_id = action_id
        self.action_name = action_name
        self.action_value = action_value
        self.cost = cost_value
        self.feature = feature_number

actions = [
    #Action('raise salary', 1.05, 7, 9),
    Action(1,'raise credit score by factor 2', 2, 9, 9),
    Action(2,'raise credit score by factor 4', 4, 9, 9)
]
NUMBER_OF_ACTIONS = len(actions)

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
        self.num_of_actions = 0
        self.num_of_successes = 0
        self.num_of_passes=0
    def createChildren(self,amount):
        for i in range (0,amount):
            self.child.append(Tree())
    def setChildrenValues(self,list):
        for i in range(0,len(list)):
            self.data.append(list[i])



def Tree_Search(root):

    #action_list = []

    if root.num_of_actions==4:
        return root.total_cost, [root.action], False
    #TODO - change the function's name after we export the loanrequest weights:
    if net.forward(root.data)==0:
       print("Loan is approved given your current financial status")
       return root.total_cost, [root.action], True

    current_action = random.choice(actions)
    #while (current_action.action_id in action_list):
    #    current_action = random.choice(actions)
    #action_list.append(current_action.action_id)
    print('Trying action:', current_action.action_name)

    root.child.append(Tree(root.data))
    child = root.child[-1]
    child.action = current_action
    child.num_of_actions = root.num_of_actions+1
    child.data[child.action.feature] = child.action.action_value * child.data[child.action.feature]

    total_cost, all_actions, is_successful = Tree_Search(child) #Call this child's child, and save the cost, the list of actions and whether the rollout was successful
    child.num_of_successes += int(is_successful) #num_of_successes contains the number of successful rollout coming out of this child
    child.num_of_passes=child.num_of_passes+1 #The number of times we have passes through this child during the run

    all_actions.append(current_action)
    child.total_cost = total_cost #Summing the cost of this child and its subtrees
    return total_cost + current_action.cost, all_actions, is_successful

df = pd.read_csv('dataset\montecarlo_trial.csv')

features_np_array = (df.iloc[0, :-1]).astype(float).to_numpy()
features_tensor = torch.from_numpy(features_np_array).type(torch.FloatTensor)
root = Tree(features_tensor)

net = loanrequest.Net(DROPOUT_RATE=0.1)
net.load_state_dict(torch.load('models\split_33_66_batchsize_500_lr_0.001_dropout_0.1_epoch_1.pkl', map_location='cpu'))

Tree_Search(root)