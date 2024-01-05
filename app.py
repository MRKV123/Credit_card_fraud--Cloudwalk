from flask import Flask, request, jsonify 
import pandas as pd
from method import dt_engineering, apply_business_rules, status_update, filter_fraudulent_transactions

app = Flask(__name__)

@app.route('/process_transaction', methods=['POST'])
def process_transaction():
    try:
        #receive the json data
        json_data = request.get_json()

        #transform the json data in dataframe
        transaction_data = pd.DataFrame([json_data], index=[0])

        #identifying fraudulent IDs
        fraudulent_ids = filter_fraudulent_transactions(transaction_data)

        #data enrichment
        transaction_data = dt_engineering(transaction_data)

        #Applying the model of score-rules
        apply_business_rules(transaction_data)

        #receive the transaction status (accept, denied)
        updated_data = status_update(transaction_data, fraudulent_ids)

        #list with transaction data and status
        result_final = updated_data[['transaction_id', 'status']]

        #transform the dataframe (result_final) to a list
        result_dict = result_final.to_dict(orient='records')

        #print the result
        return jsonify({"result": result_dict})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)




        