from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the dataset (you can adjust the path accordingly)
df = pd.read_csv('dataset.csv')

# Separate features (salary and family member) and target (zone)
X = df[['Salary', 'Member']]  # Use 'Member' here
y = df['Zone']

# Standardize features (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the K-NN classifier with a chosen value of k (e.g., k=3)
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the model on the entire dataset
knn_classifier.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    salary = float(request.form['salary'])
    family_member = int(request.form['family_member'])

    # Create a DataFrame with the user input
    user_data = pd.DataFrame({'Salary': [salary], 'Member': [family_member]})  # Use 'Member' here

    # Scale the user input data using the same scaler
    user_data_scaled = scaler.transform(user_data)

    # Find the k-nearest neighbors for the user input
    distances, indices = knn_classifier.kneighbors(user_data_scaled, n_neighbors=k)

    # Get the predicted zone for the user input
    predicted_zone = knn_classifier.predict(user_data_scaled)[0]

    # Get the zones of the k-nearest neighbors
    neighbor_zones = [y.iloc[i] for i in indices[0]]

    # Prepare data to pass to the template
    neighbors = [{'zone': zone, 'distance': distance} for zone, distance in zip(neighbor_zones, distances[0])]

    # Pass the prediction result and neighbors to the result.html template
    return render_template('result.html', predicted_zone=predicted_zone, neighbors=neighbors)

if __name__ == '__main__':
    app.run(debug=True)
