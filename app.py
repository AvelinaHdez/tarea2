from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

model = joblib.load('modelo.pkl')
scaler = joblib.load('scaler.pkl')
app.logger.debug('Modelo y Escalador cargado correctamente.')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        term = int(request.form['term'])
        credit_score = float(request.form['credit_score'])

        input_data = pd.DataFrame([[credit_score, term, income, loan_amount]], 
                                  columns=[' cibil_score', ' loan_term', ' income_annum', ' loan_amount'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)

        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {e}')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
