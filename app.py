from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the diabetes prediction model
model = pickle.load(open('diabetes.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input features from form
        try:
            features = [float(x) for x in request.form.values()]
            # Convert features into a NumPy array
            input_features = np.array(features).reshape(1, -1)
            # Make a prediction using the loaded model
            prediction = model.predict(input_features)[0]
            # Convert the prediction into a meaningful output
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        except ValueError as e:
            return render_template('index.html', error="Invalid input values. Please check your entries.")
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {e}")

        return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
