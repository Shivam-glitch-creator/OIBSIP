import joblib

# Save model to file
joblib.dump(model, "car_price_prediction_model.pkl")
print("✅ Model saved as 'car_price_prediction_model.pkl'")
# Load the saved model
loaded_model = joblib.load("car_price_prediction_model.pkl")
def predict_price():
    print("🔍 Enter Car Details:")
    present_price = float(input("Present Price (in lakhs): "))
    driven_kms = int(input("Driven Kilometers: "))
    owner = int(input("Number of Previous Owners (0/1/2+): "))
    car_age = int(input("Car Age (in years): "))
    
    fuel_type = input("Fuel Type (Petrol/Diesel): ").strip().capitalize()
    selling_type = input("Selling Type (Dealer/Individual): ").strip().capitalize()
    transmission = input("Transmission (Manual/Automatic): ").strip().capitalize()

    # Encode manually (same order as in training)
    fuel_petrol = 1 if fuel_type == "Petrol" else 0
    seller_individual = 1 if selling_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    # Create input vector
    input_data = [[present_price, driven_kms, owner, car_age,
                   fuel_petrol, seller_individual, transmission_manual]]

    # Predict
    predicted_price = model.predict(input_data)[0]
    print(f"💰 Predicted Selling Price: ₹{predicted_price:.2f} lakhs")

# Run the predictor
predict_price()
