import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from flask import Flask, render_template, request
from flask_mail import Mail, Message

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = "marimuthuari96@gmail.com"
app.config['MAIL_PASSWORD'] = "ebqrrryzrgcskjwy"

mail = Mail(app)

# Load the dataset
data = {
    'sensor1': [0.8, 0.7, 0.9, 0.1, 0.2, 0.3, 0.2, 0.1, 0.7, 0.9, 0.1, 0.2, 0.3, 0.2, 0.1],
    'sensor2': [0.5, 0.6, 0.4, 0.2, 0.3, 0.1, 0.1, 0.2, 0.6, 0.4, 0.2, 0.3, 0.1, 0.1, 0.2],
    'sensor3': [0.9, 0.8, 0.7, 0.6, 0.2, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6, 0.2, 0.8, 0.6, 0.7],
    'sensor4': [0.7, 0.6, 0.8, 0.9, 0.6, 0.5, 0.8, 0.7, 0.6, 0.8, 0.9, 0.6, 0.5, 0.8, 0.7],
    'sensor5': [0.4, 0.3, 0.5, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.5, 0.2, 0.3, 0.4, 0.5, 0.2],
    'manual11': [2, 3, 2, 1, 2, 3, 5, 2, 3, 2, 1, 2, 3, 5, 2],
    'manual12': [1, 1, 2, 0, 2, 1, 3, 2, 1, 1, 0, 2, 1, 3, 2],
    'manual13': [1, 2, 1, 0, 3, 1, 2, 1, 2, 1, 0, 3, 1, 2, 1],
    'manual14': [0, 0, 1, 1, 1, 2, 2, 3, 0, 1, 2, 1, 2, 2, 3],
    'manual15': [1, 1, 0, 2, 3, 1, 1, 0, 1, 0, 2, 1, 1, 1, 0],
    'type': ['acute', 'chronic', 'acute', 'chronic', 'acute', 'chronic', 'acute', 'chronic',
             'chronic', 'acute', 'chronic', 'acute', 'chronic', 'acute', 'chronic'],
    'diabetic_type': ['type1', 'type2', 'type3', 'type1', 'type2', 'type3', 'type2', 'type3',
                      'type2', 'type3', 'type1', 'type2', 'type3', 'type2', 'type3'],
    'Drug': ['A', 'B', 'C', 'ABC', 'B', 'C', 'A', 'A', 'B', 'C', 'ABC', 'B', 'C', 'A', 'A'],
    'amplitude': [1, 1, 0, 1, 1, 0, 2, 3, 1, 4, 5, 1, 0, 1, 1],
    'frequency': [0.2, 0.3, 0.1, 0.5, 0.6, 0.1, 1, 1, 0.9, 0.7, 0.2, 0.3, 0.1, 1, 1],
    'voltage': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.5, 0.6, 0.4, 0.3, 0.1],
    'pulse_duration': [0.1, 0.1, 0.2, 0.5, 0.6, 0.9, 0.7, 0.8, 0.9, 0.2, 0, 0.5, 0.1, 0.4, 0.2]
}
df = pd.DataFrame(data)
df[['amplitude', 'frequency', 'voltage', 'pulse_duration']] = df[['amplitude', 'frequency', 'voltage', 'pulse_duration']].astype(float)

# Split the data into input features and target
X = df.drop(['type', 'diabetic_type', 'Drug','amplitude', 'frequency', 'voltage', 'pulse_duration'], axis=1)
y = df['type']

# Build the ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
ann_model.fit(X, y)

# Build the Random Forest model for chronic wound prediction
chronic_wound_model = RandomForestClassifier(random_state=42)
chronic_wound_X = df[df['type'] == 'chronic'].drop(['type', 'diabetic_type', 'Drug','amplitude', 'frequency', 'voltage', 'pulse_duration'], axis=1)
chronic_wound_y = df[df['type'] == 'chronic']['diabetic_type']
chronic_wound_model.fit(chronic_wound_X, chronic_wound_y)
# Perform clustering for drug composition prediction
drug_X = df.drop(['type', 'diabetic_type', 'Drug','amplitude', 'frequency', 'voltage', 'pulse_duration'], axis=1)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(drug_X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_X)
le = LabelEncoder()
y = le.fit_transform(y)
# Build the XGBoost model for electrotherapy recommendation
# Build the XGBoost model for electrotherapy prediction
# Build the XGBoost model for electrotherapy recommendation
electrotherapy_model = xgb.XGBRegressor(random_state=42)
electrotherapy_X = df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5','manual11', 'manual12', 'manual13', 'manual14', 'manual15']].astype(float)
electrotherapy_y = df[['amplitude']].astype(float)
#electrotherapyf_y = df[['frequency']].astype(float)
#electrotherapyp_y = df[['pulse_duration']].astype(float)
#electrotherapyv_y = df[['voltage']].astype(float)
electrotherapy_model.fit(electrotherapy_X, electrotherapy_y)
electrotherapy_model1 = xgb.XGBRegressor(random_state=42)
electrotherapy_Xf = df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5','manual11', 'manual12', 'manual13', 'manual14', 'manual15']].astype(float)
electrotherapy_yf = df[['frequency']].astype(float)
electrotherapy_model1.fit(electrotherapy_Xf, electrotherapy_yf)

electrotherapy_model2 = xgb.XGBRegressor(random_state=42)
electrotherapy_Xp = df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5','manual11', 'manual12', 'manual13', 'manual14', 'manual15']].astype(float)
electrotherapy_yp = df[['pulse_duration']].astype(float)
electrotherapy_model2.fit(electrotherapy_Xp, electrotherapy_yp)

electrotherapy_model3 = xgb.XGBRegressor(random_state=42)
electrotherapy_Xv = df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5','manual11', 'manual12', 'manual13', 'manual14', 'manual15']].astype(float)
electrotherapy_yv = df[['voltage']].astype(float)
electrotherapy_model3.fit(electrotherapy_Xv, electrotherapy_yv)


# Define the function to send email
def send_email(recipient, subject, body):
    msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[recipient])
    msg.body = body
    mail.send(msg)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        input_data.pop('submit', None)
        input_df = pd.DataFrame([input_data])
        input_df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']] = input_df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']].astype(float)
        prediction = ann_model.predict(input_df)[0]

        # Other predictions...
        chronic_wound_prediction = chronic_wound_model.predict(input_df)[0]
        cluster_prediction = kmeans.predict(scaler.transform(input_df))[0]
        electrotherapy_prediction = electrotherapy_model.predict(input_df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'manual11', 'manual12', 'manual13', 'manual14', 'manual15']].astype(float))[0]
        electrotherapy_predictionf = electrotherapy_model1.predict(input_df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'manual11', 'manual12', 'manual13', 'manual14', 'manual15']].astype(float))[0]
        electrotherapy_predictionp = electrotherapy_model2.predict(input_df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'manual11', 'manual12', 'manual13', 'manual14', 'manual15']].astype(float))[0]
        electrotherapy_predictionv = electrotherapy_model3.predict(input_df[['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'manual11', 'manual12', 'manual13', 'manual14', 'manual15']].astype(float))[0]

        # Prepare the email content
        email_subject = "Prediction Results"
        email_body = f"ANN Prediction: {prediction}\n"
        email_body += f"Random Forest Prediction: {chronic_wound_prediction}\n"
        email_body += f"Clustering Prediction: Cluster {cluster_prediction + 1}\n"
        email_body += f"XGBoost Prediction (Amplitude): {electrotherapy_prediction}\n"
        email_body += f"XGBoost Prediction (Frequency): {electrotherapy_predictionf}\n"
        email_body += f"XGBoost Prediction (Pulse Duration): {electrotherapy_predictionp}\n"
        email_body += f"XGBoost Prediction (Voltage): {electrotherapy_predictionv}\n"

        # Send the email
        recipient_email = "marimuthuari96@gmail.com"
        send_email(recipient_email, email_subject, email_body)

        return render_template('email wound.html', prediction=prediction, chronic_wound_prediction=chronic_wound_prediction, cluster_prediction=cluster_prediction, electrotherapy_prediction=electrotherapy_prediction, electrotherapy_predictionf=electrotherapy_predictionf, electrotherapy_predictionp=electrotherapy_predictionp, electrotherapy_predictionv=electrotherapy_predictionv, prediction_sent=True)
    else:
        return render_template('email wound.html')

if __name__ == '__main__':
    app.run()
