from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('trained_random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])

        # Create input array
        input_features = np.array([[pclass, sex, age, sibsp, parch]])
        
        # Ensure the input has the correct shape
        print("Input features:", input_features)
        print("Expected number of features by model:", model.n_features_in_)

        # Prediction
        prediction = model.predict(input_features)

        return render_template('index.html', prediction_text='Survival Prediction: {}'.format(prediction[0]))

    except Exception as e:
        print("Error:", e)
        return render_template('index.html', prediction_text='Error: Incorrect number of features')

if __name__ == "__main__":
    app.run(debug=True)
