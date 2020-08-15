import numpy as np
import pandas as pd
import preprocessor
import loanrequest
import main
import montecarlo
import torch
from flask import Flask, request
app = Flask(__name__)

@app.route("/test_candidate.py")
def hello():
    print('hello called')
    origination_channel = request.args.get('origination_channel', type=float)
    seller_name = request.args.get('seller_name', type=float)
    interest_rate = request.args.get('interest_rate', type=float)
    upb = request.args.get('upb', type=float)
    orig_loan_t = request.args.get('orig_loan_t', type=float)
    total_price = request.args.get('total_price', type=float)
    first_lien = request.args.get('first_lien', type=float)
    current_amount = request.args.get('current_amount', type=float)
    second_amount = request.args.get('second_amount', type=float)
    num_borr = request.args.get('num_borr', type=float)
    monthly_payments = request.args.get('monthly_payments', type=float)
    income = request.args.get('income', type=float)
    borrower_credit_score = request.args.get('borrower_credit_score', type=float)
    first_time = request.args.get('first_time', type=float)
    loan_purp = request.args.get('loan_purp', type=float)
    num_units = request.args.get('num_units', type=float)
    occ_type = request.args.get('occ_type', type=float)
    zip = request.args.get('zip', type=float)
    co_credit_score = request.args.get('co_credit_score', type=float)
    ins_perc = request.args.get('ins_perc', type=float)
    ins_type = request.args.get('ins_type', type=float)
    reloc_ind = request.args.get('reloc_ind', type=float)
    state = request.args.get('state', type=float)

    stats = pd.read_csv('dataset\statistics.csv')
    stats_mean = stats.iloc[1]
    stats_std = stats.iloc[2]

    # Other parameters we need to calculate:
    ltv = upb / total_price * 100
    cltv = ( first_lien + current_amount + second_amount ) / total_price * 100
    dti = monthly_payments / income * 100
    if interest_rate == -1:
        interest_rate = stats_mean[3]

    # Build feature list for new person
    new_person = np.array([origination_channel, seller_name, interest_rate, upb, orig_loan_t, ltv, cltv,
                           num_borr, dti, borrower_credit_score, first_time, loan_purp, num_units, occ_type, 1.0, state, zip,
                           ins_perc, co_credit_score, ins_type, reloc_ind])

    # Instansiate net
    net = main.Net(DROPOUT_RATE=0.0001)
    net.load_state_dict(
        torch.load('models/final_weights.pkl', map_location='cpu'))

    # Call the net with the person's online data
    new_person_np = new_person
    new_person = pd.DataFrame(new_person).transpose().astype(float)
    new_person_normalized = preprocessor.norm_features(new_person, montecarlo.stats)
    new_person_normalized = torch.tensor(new_person_normalized.values).float()
    net_out = float(net.forward(new_person_normalized))

    # Either tell the client his application will be approved, or call monte-carlo code to suggest
    print(net_out)
    if net_out < 0.4:
        print("Mortgage request is approved")
        return str("Mortgage request is approved")
    else:
        print("Mortgage request is declined")
        features_tensor = torch.from_numpy(new_person_np).type(torch.FloatTensor)
        root = montecarlo.Tree(features_tensor)
        root.action = montecarlo.Action(action_id=0, action_name="current_state", action_value=0, cost_value=0, feature_number=0)  # We create a fictive action for the root, just to make sure the algorithm runs well. We will delete this from the proposed list.
        plan = montecarlo.monte_carlo_tree_search(root)
        return plan

if __name__ == "__main__":
    app.run()
