import requests

url = 'http://127.0.0.1:5000/predict_api'  # URL of the running Flask app

# Example input data for prediction
data = {
    'pclass': 3,
    'sex': 1,  # 0 = female, 1 = male
    'age': 22,
    'sibsp': 1,  # siblings
    'parch': 0,  # parents
}

# Send POST request to the API
response = requests.post(url, json=data)

# Print the response
print('Predicted result:', response.json())
