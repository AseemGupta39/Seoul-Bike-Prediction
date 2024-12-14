import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 
import os
from datetime import datetime

class Inference:
    def __init__(self,model_path,scaler_model_path):
        self.model_path = model_path
        self.scaler_model_path = scaler_model_path

        if os.path.exists(self.model_path) and os.path.exists(self.scaler_model_path):
            self.model = pickle.load(open(self.model_path,'rb'))
            self.sc = pickle.load(open(self.scaler_model_path,'rb'))
            print(self.model,self.sc,sep = "\n\n",end = "\n\n")
        else:
            print("Model file path is incorrect")
    
    @staticmethod
    def get_string_to_datetime(date):
        dt = datetime.strptime(date, "%d/%m/%Y")
        return {
            "day": dt.day, "month": dt.month,\
            "year": dt.year, "week_day": dt.strftime("%A")
            }

    @staticmethod
    def season_to_df(season):
        seasons_cols = ['Spring', 'Summer', 'Winter']
        seasons_data = [[0 for _ in range(len(seasons_cols))]]
        df_seasons = pd.DataFrame(seasons_data,columns=seasons_cols)
        if season in seasons_cols:
            df_seasons[season] = 1
        return df_seasons
    
    @staticmethod
    def day_to_df(week_day):
        day_names = ['Monday', 'Saturday','Sunday', 'Thursday', 'Tuesday', 'Wednesday']
        day_names_data =[[0 for _ in range(len(day_names))]]
        df_days = pd.DataFrame(day_names_data,columns=day_names)

        if week_day in day_names:
            df_days[week_day] = 1
        
        return df_days


    def users_input(self):
        print("Enter correct information to predict Rented Bile count for a day with the following details:")

        date = input("Date (dd/mm/yyyy): ")
        hour = int(input("Hours (0-23): "))
        temperature = float(input("Temperature in C: "))
        humidity = float(input("Humidity %: "))
        wind_speed = float(input("Wind Speed (m/s): "))
        visibility = float(input("Visibility (in 10m): "))
        solar_radiation = float(input("Solar Radiation (MJ/m2): "))
        rainfall = float(input("Rainfall (mm): "))
        snowfall = float(input("Snowfall (cm): "))
        seasons = input("Season (Autumn, Spring, Summer, Winter): ")
        holiday = input("Holiday (Holiday/No Holiday): ")
        functioning_day = input("Functioning Day (Yes/No): ")
        
        
        holiday_dic = {"No Holiday": 0, "Holiday": 1}
        functioning_dic = {"No": 0, "Yes": 1}

        str_to_date = self.get_string_to_datetime(date)

        u_input_list = [
            hour, temperature, humidity,\
            wind_speed, visibility, solar_radiation,\
            rainfall, snowfall,holiday_dic[holiday],\
            functioning_dic[functioning_day],str_to_date["day"],\
            str_to_date["month"], str_to_date["year"]
            ]

        features_name = [
            "Hour", "Temperature(°C)", "Humidity(%)",\
            "Wind speed (m/s)", "Visibility (10m)",\
            "Solar Radiation (MJ/m2)", "Rainfall(mm)",\
            "Snowfall (cm)", "Holiday", "Functioning Day",\
            "Day", "Month", "Year"
            ]

        df_user_input = pd.DataFrame([u_input_list],columns=features_name)
        df_season = self.season_to_df(seasons)
        df_days = self.day_to_df(str_to_date['week_day'])

        df_for_pred = pd.concat([df_user_input,df_season,df_days],axis = 1)
        
        return df_for_pred
    
    def predict(self):
        df = self.users_input()
        scaled_data = self.sc.transform(df)
        prediction = self.model.predict(scaled_data)
        return round(prediction[0])
    
if __name__ == '__main__':
    ml_model_path = r'models\XGBRegressor_r2_0_952_v1.pkl'
    scaler_model_path = r'models\sc.pkl'

    inference = Inference(ml_model_path,scaler_model_path)

    prediction = inference.predict()

    print(f"Count of bikes required would be as per the user factors is: {prediction}")

            