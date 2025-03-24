import numpy as np
import joblib
import sys
import time
import os

# =========================================
# 1. Load Trained Model and Preprocessors
# =========================================

# Base directory where your files are stored
try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # doesn't work well in IDE's
except NameError:
        BASE_DIR = os.getcwd() # Grabs current working direction as a str

# Full paths to saved model objects
MODEL_PATH = os.path.join(BASE_DIR, 'satcom_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'satcom_scaler.pkl')
POLY_PATH = os.path.join(BASE_DIR, 'satcom_poly.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    poly = joblib.load(POLY_PATH)
    print("Model and preprocessors loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model or preprocessors: {e}")
    sys.exit(1)

# =========================================
# 2. User Input Function (With Validations)
# =========================================
def get_user_input():
    print("\nEnter the SATCOM parameters (type 'exit' to quit):")

    while True:
        try:
            raw_input_power = input("Input Power (dBm) [0 to 30]: ")
            if raw_input_power.lower() == 'exit':
                sys.exit(0)
            input_power = float(raw_input_power)

            if input_power < 0 or input_power > 30:
                print("Caution: Input Power is outside the general recommended limits (0-30 dBm)."
                + " Proceeding anyway.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for Input Power.")
            continue

        try:
            raw_attenuation = input("Attenuation (dB) [0 to 28]: ")
            if raw_attenuation.lower() == 'exit':
                sys.exit(0)
            attenuation = float(raw_attenuation)

            if attenuation < 0 or attenuation > 28:
                print("Caution: Attenuation is outside the general recommended limits (0-28 dB)."
                + " Proceeding anyway.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for Attenuation.")
            continue

        return input_power, attenuation

# =========================================
# 3. Prediction Function (With Fixed HPA Gain of 30dB)
# =========================================
def predict_output_power(input_power, attenuation, fixed_hpa_gain=30):
    """
    Predict output power based on user input and trained AI model.
    """
    try:
        user_input = np.array([[input_power, attenuation, fixed_hpa_gain]])

        user_input_scaled = scaler.transform(user_input)
        user_input_poly = poly.transform(user_input_scaled)

        predicted_output = model.predict(user_input_poly)


        return predicted_output[0]

    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

# =========================================
# 4. Logging Function (Optional)
# =========================================
#def log_prediction(input_power, attenuation, predicted_power, filename='prediction_log.csv'):
    """
    Logs predictions to a CSV file for later analysis.
    """
    with open(filename, 'a') as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{input_power},{attenuation},{predicted_power:.2f}\n")

# =========================================
# 5. Main Execution Loop
# =========================================
def main():
    PSAT = 50

    print("SATCOM Output Power Prediction Tool\n")
    print("Type 'exit' at any time to quit.\n")

    while True:
        input_power, attenuation = get_user_input()

        predicted_power = predict_output_power(input_power, attenuation)

        if predicted_power is not None:
            print("\nPrediction Complete!")
            print(f"Input Power (dBm): {input_power}")
            print(f"Attenuation (dB): {attenuation}")
            if predicted_power < PSAT:
                print(f"Predicted Output Power (dBm): {predicted_power:.2f}\n")
            else:
                predicted_power = predicted_power - PSAT
                print(f"Your parameters exceded the Psat by: {predicted_power:.2f} dBm")
                print(f"Limiting your output power to the saturation point...")

                time.sleep(3)
                print(f"Predicted Output Power (dBm): {PSAT:.2f}\n")

            # Optional logging
            #log_prediction(input_power, attenuation, predicted_power)
        else:
            print("Prediction returned no value.\n")

# =========================================
# 6. Run the Script
# =========================================
if __name__ == '__main__':
    main()
