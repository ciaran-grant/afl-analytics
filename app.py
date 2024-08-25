import numpy as np
import pandas as pd
import warnings
from flask import Flask, request
from AFLPy.AFLData_Client import load_data, upload_data
from AFLPy.ntfy import push_notification
from afl_analytics.arpadl.pyafl import convert_to_actions

warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route("/aflanalytics/convert_to_actions", methods=["GET", "POST"])
def convert_chains_to_arpadl(ID = None):
    
    chains = load_data('AFL_API_Match_Chains', ID = request.json['ID'])
    actions = convert_to_actions(chains)
    
    upload_data(Dataset_Name="CG_ARPADL_Data", Dataset=actions, overwrite=True, update_if_identical=True)
    push_notification("Chains converted to ARPADL Data", ", ".join([match_id for match_id in list(actions['match_id'])]))

    return actions.to_json(orient='records')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8005, debug=False)
    