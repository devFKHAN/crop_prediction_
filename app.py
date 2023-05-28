from flask import Flask, request, jsonify
import numpy as np
import pickle
model = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    n = request.form.get('Nitrogen')
    p = request.form.get('Phosphorus')
    k = request.form.get('Potassium')
    ph = request.form.get('pH')
    mos = request.form.get('Moisture')
    temp = request.form.get('Temperature')
    humidity = request.form.get('humidity')
    input_query = np.array([[n, p, k, ph, mos, temp, humidity]])

    result = model.predict(input_query)[0]

    return jsonify({'result': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
