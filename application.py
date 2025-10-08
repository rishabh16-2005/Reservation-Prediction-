import joblib
import numpy as np 
from config.path_config import MODEL_OUTPUT_PATH
from flask import Flask , render_template,request

app = Flask(__name__)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        room_type_reserved = int(request.form['room_type_reserved'])
        required_car_parking_space = int(request.form['required_car_parking_space'])
        market_segment_type = int(request.form['market_segment_type'])
        type_of_meal_plan = int(request.form['type_of_meal_plan'])
        lead_time = int(request.form['lead_time'])
        no_of_special_requests = int(request.form['no_of_special_requests'])
        avg_price_per_room = float(request.form['avg_price_per_room'])
        no_of_week_nights = int(request.form['no_of_week_nights'])
        no_of_adults = int(request.form['no_of_adults'])
        no_of_weekend_nights = int(request.form['no_of_weekend_nights'])
        no_of_children = int(request.form['no_of_children'])
        no_of_previous_bookings_not_canceled = int(request.form['no_of_previous_bookings_not_canceled'])
        no_of_previous_cancellations = int(request.form['no_of_previous_cancellations'])
        features = np.array([[room_type_reserved,required_car_parking_space,market_segment_type,type_of_meal_plan,lead_time,no_of_special_requests,avg_price_per_room,no_of_week_nights,no_of_adults,no_of_weekend_nights,no_of_children,no_of_previous_bookings_not_canceled,no_of_previous_cancellations]])

        prediction = loaded_model.predict(features)

        return render_template('index.html',prediction=prediction[0])
    return render_template('index.html',prediction=None)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)