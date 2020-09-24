import random
import main
import torch
import pandas as pd
import math
import statistics
import copy
import numpy as np
import preprocessor
import matplotlib.pyplot
from visualize_tree import visualize_tree

# For a given user, we want deterministic results
random.seed()

# Hyper Parameters
MONTE_CARLO_ITERS_NUM = 1000000 # Number of monte-carlo iteration, where each iteration is made of up selection, expansion, simulation and backpropogaition
NET_THRESHOLD = 0.5 # The threshold of the network's output for which we determine it changed its decision
EXPANSION_MAX_DEPTH = 3  # The maximal depth of the real built tree
SIMULATION_MAX_DEPTH = 3  # The maximal total depth of the simulated tree
A = 1 # The constant that measures the weight we give to the number of successes when traveling the tree
B = math.sqrt(2) # The constant that measures the weight we give to the number of visits
C = 2 # The constant that measures the weight we give to the cost of the chosen actions
D = math.sqrt(2) # The constant that measures the weight we give to the tree's depth

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

    scores = np.array([])

    for i in range(MONTE_CARLO_ITERS_NUM):

        # Selection stage
        path_to_leaf = selection(root)
        if len(path_to_leaf) > 1:
            leaf = path_to_leaf[-1]
        else:
            leaf = path_to_leaf[0]

        # Expansion Stage
        if leaf.depth != EXPANSION_MAX_DEPTH:
            next_child = expansion(leaf)
        else:
            next_child = leaf

        next_child_sim = copy.copy(next_child)

        # Simulation Stage
        total_cost, all_actions, is_successful = simulation(next_child_sim)

        # Back propogation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        path_to_leaf.append(next_child)
        backpropogation(path_to_leaf, is_successful)

        # Look at the current score of the best path
        _, best_score = best_route(root)
        scores = np.append(scores, best_score)
        print(i, best_score)

    matplotlib.pyplot.plot(scores[1000:])
    matplotlib.pyplot.show()

    #Choose the best route of all and propose it to the user:
    proposed_actions, _ = best_route(root)

    proposed_actions = proposed_actions[:-1]

    if proposed_actions==[]:
        return str("No reasonable changes to help your application become approved have been found.")

    return str("In order to make your mortgage application approved: " + str(proposed_actions))


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

    # Per Ronen's request, we examine all of the actions, and not a random subset
    for action in actions_list:
        leaf.child.append(Tree(leaf.data))  # Add child to the current node
        child = leaf.child[-1]
        child.action = action
        child.depth = leaf.depth + 1
        child.total_cost = leaf.total_cost + action.cost
        child.data[child.action.feature] = child.action.action_value * child.data[child.action.feature]

    # Pick random child to go down
    return random.choice(leaf.child)


def simulation(node):

    if node.depth == SIMULATION_MAX_DEPTH:
        return node.total_cost, [node.action], False

    # Tensor to dataframe
    df_data = pd.DataFrame(data=node.data.numpy().copy()).T
    # Normalize
    norm_data = preprocessor.norm_features(df_data, stats)
    # Dataframe to tensor
    norm_data = torch.tensor(norm_data.values).float()
    net_out = float(net.forward(norm_data))

    if net_out <= NET_THRESHOLD:
       return node.total_cost, [node.action], True

    key_list = list(actions.keys())
    key = random.choice(key_list)
    current_action = random.choice(actions[key])

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
    if node.child == []:  #if it's a successful leaf: meaning there are no children
        temp.append(node.action.action_name)
        return (temp, 0)

    list_of_actions = []

    max_score = -math.inf
    max_score_idx = 1.5
    for i, child in enumerate(node.child):
        if child.num_of_successes == 0:
            continue
        score = A * child.num_of_successes / (child.num_of_passes + 1)
        score -= C * child.total_cost / (MEAN_ACTION_COST * child.depth)
        score -= D * child.depth
        if score > max_score:
            max_score = score
            max_score_idx = i

    # If this node has no successful children at all, just add the current action
    # and return
    if max_score_idx == 1.5:
        list_of_actions.append(node.action.action_name)
        return (list_of_actions, 0)

    rest_of_path, rest_of_score = best_route(node.child[max_score_idx])
    list_of_actions.extend(rest_of_path)
    list_of_actions.append(node.action.action_name)
    return (list_of_actions, (max_score + rest_of_score))


def generate_actions (feature,values,curr_value, is_discrete):
    actions = []
    for i in values:
        if is_discrete:
            action_value = i / curr_value.iloc[0,feature]
            action_name = 'change feature ' + str(feature) + ' to ' + str(i)
            if i == curr_value.iloc[0,feature]:
                continue
        else:
            action_value = i
            action_name = str('multiply feature ') + str(feature) + str(' by ') + str(action_value)
        curr_cost = (abs(action_value * curr_value.iloc[0,feature] - stats_mean.iloc[feature+1])) / stats_std.iloc[feature+1]
        TempAction = Action(str(feature) + str(i), create_action_name(feature,i),
                            action_value, curr_cost, feature)
        actions.append(TempAction)
    return actions


def create_action_name(feature, value):

    if feature == 0:
        if value == 1:
            new_origination_channel = str('"Retail"')
        elif value == 2:
            new_origination_channel = str('"Correspondent"')
        else:
            new_origination_channel = str('"Broker"')
        return str("Change your origination channel to " + new_origination_channel)

    if feature == 1:
        seller_names = ['WELLS FARGO BANK,  NA', 'AMERIHOME MORTGAGE COMPANY, LLC', 'METLIFE HOME LOANS LLC',
                        'SANTANDER BANK, NATIONAL ASSOCIATION', 'PACIFIC UNION FINANCIAL, LLC', 'CASHCALL, INC.',
                        'PULTE MORTGAGE, L.L.C.', 'CMG MORTGAGE, INC', 'GMAC MORTGAGE, LLC',
                        'CAPITAL ONE, NATIONAL ASSOCIATION', 'USAA FEDERAL SAVINGS BANK',
                        'FIRST BANK DBA FIRST BANK MORTGAGE', 'LAKEVIEW LOAN SERVICING, LLC', 'FLAGSTAR BANK, FSB',
                        'PMT CREDIT RISK TRANSFER TRUST 2015-2', 'FDIC, RECEIVER, INDYMAC FEDERAL BANK FSB',
                        'CITIMORTGAGE, INC.', 'SUNTRUST MORTGAGE INC.', 'REGIONS BANK',
                        'HSBC BANK USA, NATIONAL ASSOCIATION', 'STONEGATE MORTGAGE CORPORATION', 'PMTT4',
                        'TRUIST BANK (FORMERLY SUNTRUST BANK)',
                        'CHICAGO MORTGAGE SOLUTIONS DBA INTERBANK MORTGAGE COMPANY', 'RBC MORTGAGE COMPANY',
                        'NYCB MORTGAGE COMPANY, LLC', 'FRANKLIN AMERICAN MORTGAGE COMPANY',
                        'THE BRANCH BANKING AND TRUST COMPANY',
                        'UNITED SHORE FINANCIAL SERVICES, LLC D/B/A UNITED WHOLESALE MORTGAGE',
                        'HOMEWARD RESIDENTIAL, INC.', 'NETBANK FUNDING SERVICES', 'COLORADO FEDERAL SAVINGS BANK',
                        'FREMONT BANK', 'PHH MORTGAGE CORPORATION (USAA FEDERAL SAVINGS BANK)',
                        'HOMEBRIDGE FINANCIAL SERVICES, INC.', 'SIERRA PACIFIC MORTGAGE COMPANY, INC.',
                        'FEDERAL HOME LOAN BANK OF CHICAGO', 'PROSPECT MORTGAGE, LLC', 'ASSOCIATED BANK, NA',
                        'PMT CREDIT RISK TRANSFER TRUST 2016-1', 'JPMORGAN CHASE BANK, NATIONAL ASSOCIATION',
                        'AMTRUST BANK', 'JPMORGAN CHASE BANK, NA',
                        'PRINCIPAL RESIDENTIAL MORTGAGE CAPITAL RESOURCES, LLC',
                        'GMAC MORTGAGE, LLC (USAA FEDERAL SAVINGS BANK)', 'U.S. BANK N.A.',
                        'BISHOPS GATE RESIDENTIAL MORTGAGE TRUST', 'GUILD MORTGAGE COMPANY', 'OTHER',
                        'EAGLE HOME MORTGAGE, LLC', 'WELLS FARGO CREDIT RISK TRANSFER SECURITIES TRUST 2015',
                        'EVERBANK', 'FAIRWAY INDEPENDENT MORTGAGE CORPORATION', 'ROUNDPOINT MORTGAGE COMPANY',
                        'THIRD FEDERAL SAVINGS AND LOAN', 'SUNTRUST BANK', 'NATIONSTAR MORTGAGE, LLC', 'PNC BANK, N.A.',
                        'METLIFE BANK, NA', 'J.P. MORGAN MADISON AVENUE SECURITIES TRUST, SERIES 2015-1',
                        'FLAGSTAR CAPITAL MARKETS CORPORATION', 'IMPAC MORTGAGE CORP.',
                        'UNITED SHORE FINANCIAL SERVICES, LLC DBA UNITED WHOLESALE MORTGAGE', 'LOANDEPOT.COM, LLC',
                        'ALLY BANK', 'QUICKEN LOANS INC.', 'THE HUNTINGTON NATIONAL BANK',
                        'CHICAGO MORTGAGE SOLUTIONS DBA INTERFIRST MORTGAGE COMPANY', 'WELLS FARGO BANK, N.A.',
                        'J.P. MORGAN MADISON AVENUE SECURITIES TRUST, SERIES 2014-1', 'DITECH FINANCIAL LLC',
                        'BANK OF AMERICA, N.A.', 'CHASE HOME FINANCE, LLC', 'CHASE HOME FINANCE',
                        'CHASE HOME FINANCE (CIE 1)', 'AMERISAVE MORTGAGE CORPORATION', 'MOVEMENT MORTGAGE, LLC',
                        'FIRST TENNESSEE BANK NATIONAL ASSOCIATION', 'FINANCE OF AMERICA MORTGAGE LLC',
                        'PENNYMAC CORP.', 'CHASE HOME FINANCE FRANKLIN AMERICAN MORTGAGE COMPANY',
                        'WITMER FUNDING, LLC', 'JP MORGAN CHASE BANK, NA', 'IRWIN MORTGAGE, CORPORATION',
                        'USAA DIRECT DELIVERY', 'CALIBER HOME LOANS, INC.', 'DOWNEY SAVINGS AND LOAN ASSOCIATION, F.A.',
                        'FLEET NATIONAL BANK', 'FREEDOM MORTGAGE CORP.', 'STEARNS LENDING, LLC',
                        'HARWOOD STREET FUNDING I, LLC', 'CITIZENS BANK, NATIONAL ASSOCIATION',
                        'NEW YORK COMMUNITY BANK', 'PHH MORTGAGE CORPORATION', 'FIFTH THIRD BANK',
                        'PROVIDENT FUNDING ASSOCIATES, L.P.']
        new_bank = seller_names[int(value) - 1]
        return str("Request from a different bank:" + str(new_bank))

    if feature == 4:
        return str("Change your UPB to " + str(value*features_np_array[feature]))

    if feature == 5:
        return str("Change your LTV to " + str(value*features_np_array[feature]))

    if feature == 6:
        return str("Change your CLTV to " + str(value*features_np_array[feature]))

    if feature == 7:
        return str("Change the number of borrowers to " + str(value))

    if feature == 8:
        return str("Change your debt-to-income ratio to " + str(value * features_np_array[feature]))

    if feature == 9:
        return str("Change your credit score ratio to " + str(value * features_np_array[feature]))

    if feature == 10:
        if value == 1:
            return str("Request through a non-first-time home buyer")
        if value == 2:
            return str("Request through a first-time home buyer")

    if feature == 11:
        if value == 1:
            return str("Request as a purchase-purposed loan (and not a refinance-purposed loan, for example")
        if value == 2:
            return str("Request as a cash-out refinance loan")
        if value == 3:
            return str("Request as a non-cash-out refinance loan")
        if value == 4:
            return str("Request a general refinance loan without declaring whether it's for cash-out")

    if feature == 12:
        if value == 1:
            return str("Request a property which consists of 1 unit")
        else:
            return str("Request a property which consists of " + str(value) + " units")

    if feature == 13:
        if value == 1:
            return str("Request as your principal property")
        if value == 2:
            return str("Request as your second property")
        if value == 3:
            return str("Request as an investor")
        if value == 4:
            return str("Request without declaring whether it's your principal or second property or you're an investor")

    if feature == 18:
        return str("Change your co-borrower credit score to " + str(value*features_np_array[feature]))

    if feature == 19:
        if value == 1:
            return str("Get a borrower-paid mortgage insurance")
        if value == 2:
            return str("Get a lender-paid mortgage insurance")
        if value == 3:
            return str("Get an investor-paid mortgage insurance")
        if value == 4:
            return str("Cancel your mortgage insurance")

    if feature == 20:
        if value == 1:
            return str("Request as a non-relocation property")
        if value == 2:
            return str("Request as a property for relocation")


#Load the statistics about the data
stats = pd.read_csv('dataset\statistics.csv')
stats_mean = stats.iloc[1]
stats_std = stats.iloc[2]

# Load request data and tokenize
df = pd.read_csv('dataset\montecarlo_trial.csv', names=preprocessor.COL_NAMES)
df = preprocessor.prep_columns(df)
features_np_array = (df.iloc[0, :-1]).astype(float).to_numpy()

#Generating actions for each feature:
actions = {} ##Initializing an empty dictionary of actions, where the key is the feature number it affects, and the value is a list of actions
actions[0] = (generate_actions(0,[1,2,3],df,True))  # Origination channel
actions[1] = (generate_actions(1,list(range(1,97)),df,True))  # Seller name
actions[4] = (generate_actions(4,list(np.arange(999,500,-99)/1000),df,False)) # UPB - Decrease by up to 50%
actions[5] = (generate_actions(5,list(np.arange(999,500,-99)/1000),df,False)) # LTV - Decrease by up to 50%
actions[6] = (generate_actions(6,list(np.arange(999,500,-99)/1000),df,False))  # CLTV
actions[7] = (generate_actions(7,[1,2,3],df,True)) ##Number of borrowers
actions[8] = (generate_actions(8,list(np.arange(999,500,-99)/1000),df,False))  # Debt to income
actions[9] = (generate_actions(9,list(np.arange(1001,1500, 99)/1000),df,False))  # Credit Score
actions[10] = (generate_actions(10,[1,2],df,True)) #First time home buyer
actions[11] = (generate_actions(11,list(range(1,4)),df,True)) #LOAN PURPOSE
actions[12] = (generate_actions(12,list(range(1,5)),df,True)) # Number of units
actions[13] = (generate_actions(13,list(range(1,4)),df,True)) # Occupancy Type
actions[16] = (generate_actions(16,list(np.arange(1001,1500, 99)/1000),df,False))  # PRIMARY MORTGAGE INSURANCE PERCENT
actions[18] = (generate_actions(18,list(np.arange(1001,1500, 99)/1000),df,False))  # CoBorrower Credit Score
actions[19] = (generate_actions(19,list(range(1,4)),df,True)) #MORTGAGE INSURANCE TYPE
actions[20] = (generate_actions(20,list(range(1,3)),df,True)) #RELOCATION MORTGAGE INDICATOR

# Flattened list of actions
actions_list = []
for key in actions:
    actions_list.extend(actions[key])
MEAN_ACTION_COST = statistics.mean([action.cost for action in actions_list]) # List Comprehension - create a list of only the costs of all of the actions

features_tensor = torch.from_numpy(features_np_array).type(torch.FloatTensor)
root = Tree(features_tensor)
root.action = Action(action_id=0, action_name="current_state", action_value=0, cost_value=0, feature_number=0) #We create a fictive action for the root, just to make sure the algorithm runs well. We will delete this from the proposed list.

net = main.Net(DROPOUT_RATE=0.1)
net.load_state_dict(torch.load('models/final_weights.pkl', map_location='cpu'))

res = monte_carlo_tree_search(root)