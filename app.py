#  Import Libraries
from flask import Flask, render_template, request
import joblib
import pandas as pd

#  Initialize Flask App
app = Flask(__name__)

#  Load the trained model
model = joblib.load('rf_model.pkl')

#  Define expected columns (same as training)
COLUMNS = [
    'number of adults', 'number of children', 'number of weekend nights', 'number of week nights',
    'car parking space', 'lead time', 'repeated', 'P-C', 'P-not-C', 'average price',
    'special requests', 'total_guests',
    'type of meal_Meal Plan 2', 'type of meal_Not Selected',
    'room type_Room_Type 2', 'room type_Room_Type 3', 'room type_Room_Type 4',
    'room type_Room_Type 5', 'room type_Room_Type 6', 'room type_Room_Type 7',
    'market segment type_Complementary', 'market segment type_Corporate',
    'market segment type_Offline', 'market segment type_Online'
]

#  Home page (form)
@app.route('/')
def home():
    return render_template('index.html')

#  Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        #  Get form data
        form_data = request.form.to_dict()

        #  Numeric inputs
        adults = int(form_data['number_of_adults'])
        children = int(form_data['number_of_children'])
        weekend_nights = int(form_data['number_of_weekend_nights'])
        week_nights = int(form_data['number_of_week_nights'])
        parking = int(form_data['car_parking_space'])
        lead_time = int(form_data['lead_time'])
        average_price = float(form_data['average_price'])
        special_requests = int(form_data['special_requests'])
        repeated = int(form_data['repeated'])
        pc = int(form_data['P_C'])
        pnotc = int(form_data['P_not_C'])

        #  Feature Engineering
        total_guests = adults + children

        #  One-hot encoding for categorical inputs
        meal_plan = form_data['type_of_meal']
        room_type = form_data['room_type']
        market_segment = form_data['market_segment_type']

        data = {
            'number of adults': adults,
            'number of children': children,
            'number of weekend nights': weekend_nights,
            'number of week nights': week_nights,
            'car parking space': parking,
            'lead time': lead_time,
            'repeated': repeated,
            'P-C': pc,
            'P-not-C': pnotc,
            'average price': average_price,
            'special requests': special_requests,
            'total_guests': total_guests,
        }

        # One-hot meal
        data.update({
            'type of meal_Meal Plan 2': 1 if meal_plan == 'Meal Plan 2' else 0,
            'type of meal_Not Selected': 1 if meal_plan == 'Not Selected' else 0
        })

        # One-hot room
        for i in range(2, 8):
            data[f'room type_Room_Type {i}'] = 1 if room_type == f'Room_Type {i}' else 0

        # One-hot market segment
        data.update({
            'market segment type_Complementary': 1 if market_segment == 'Complementary' else 0,
            'market segment type_Corporate': 1 if market_segment == 'Corporate' else 0,
            'market segment type_Offline': 1 if market_segment == 'Offline' else 0,
            'market segment type_Online': 1 if market_segment == 'Online' else 0
        })

        #  Convert to DataFrame
        df = pd.DataFrame([data], columns=COLUMNS)

        #  Make Prediction
        prediction = model.predict(df)[0]
        result = 'Canceled' if prediction == 1 else 'Not Canceled'

        return render_template('result.html', prediction=result)

#  Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
