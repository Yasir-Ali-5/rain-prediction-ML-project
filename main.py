from scr.Data_Loading import load_data
from scr.Data_Cleaning import clean_data
from scr.Data_Spliting import split_data
from scr.Data_Transformation import encode_data
from pipeline.Train import train_model, save_model
from pipeline.evaluate import get_score

data = load_data("artifacts\weatherAUS.csv")
print("Data loading is done@@@@@@")

cleaned_data = clean_data(data)
print("Data cleaning is done@@@@@@")

encode_data(cleaned_data,"Location")
encode_data(cleaned_data,"RainToday")
encode_data(cleaned_data,"RainTomorrow")
print("Data encoding is done @@@@@@")

X_train, X_test, y_train, y_test = split_data(cleaned_data)
print("Data Spliting is done @@@@@@")

model = train_model(X_train, y_train, X_test, y_test)
print("Data model building is done @@@@@@")

best_model = max(model, key=model.get)
save_model(best_model)
print(f"the best model {best_model} that is saved")

