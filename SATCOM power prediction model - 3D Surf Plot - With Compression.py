#=====================================README=====================================
"""
    Updated SATCOM HPA Gain Compression Model with Logistic Curve:
    - Simplified visualization to focus on Effective Input Power vs Output Power.
    - Logistic compression curve replaces previous methods.
"""
#=====================================README=====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.interpolate import bisplrep, bisplev
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from mpl_toolkits.mplot3d import Axes3D
import os
import joblib

# 0. Compression and Saturation Function (Logistic)
def logistic_compression(input_power, gain=30, p1dB=46, psat=50, smoothness=0.5):
    """
    Logistic compression curve with input/output clipping for stability.
    """
    linear_output = input_power + gain
    
    # Clip to avoid overflow/underflow in exp
    linear_output_clipped = np.clip(linear_output, p1dB - 20, p1dB + 20)
    
    compressed_output = psat / (1 + np.exp(-smoothness * (linear_output_clipped - p1dB)))

    # Ensure final output doesn't exceed Psat
    return np.clip(compressed_output, None, psat)

# 1. Generate Simulated SATCOM Dataset
def generate_satcom_data(samples=1000, noise_level=0.5, fixed_hpa_gain=30, p1dB=47, psat=50):
    """
    Generate synthetic SATCOM data, including forced zero attenuation samples for regression.
    """
    input_power = np.random.uniform(5, 28, samples)
    attenuation = np.random.uniform(0, 28, samples)

    zero_attenuation_samples = int(samples * 0.1)
    if zero_attenuation_samples > 0:
        attenuation[:zero_attenuation_samples] = 0
        np.random.shuffle(attenuation)

    noise = np.random.normal(0, noise_level, samples)

    effective_input_power = input_power - attenuation
    linear_output_power = effective_input_power + fixed_hpa_gain + noise

    output_power = logistic_compression(
        effective_input_power,
        gain=fixed_hpa_gain,
        p1dB=p1dB,
        psat=psat,
        smoothness=0.5
    )

    df = pd.DataFrame({
        'Input_Power_dBm': input_power,
        'Attenuation_D_B': attenuation,
        'Effective_Input_Power_dBm': effective_input_power,
        'HPA_Gain_dB': np.full(samples, fixed_hpa_gain),
        'Linear_Output_Power_dBm': linear_output_power,
        'Output_Power_dBm': output_power
    })

    return df

# 2. Preprocessing Function
def preprocess_data(df, degree=2):
    X = df[['Input_Power_dBm', 'Attenuation_D_B', 'HPA_Gain_dB']].values
    y = df['Output_Power_dBm'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_scaled)

    return X_poly, y, scaler, poly

# 3. Train the Model
def train_model(X_poly, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)
    return model

# 4. Evaluate the Model
def evaluate_model(model, X_poly, y, cv_folds=5):
    y_pred = model.predict(X_poly)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    cv_scores = cross_val_score(model, X_poly, y, cv=cv_folds, scoring='r2')

    print(f"Mean Squared Error: {mse:.3f}")
    print(f"R² Score (train): {r2:.3f}")
    print(f"Cross-Validated R² Scores: {cv_scores}")
    print(f"Mean CV R² Score: {cv_scores.mean():.3f}")

    return y_pred

# 5. Visualization Function (Updated)
def visualize_model(df, model, scaler, poly, fixed_hpa_gain=30, p1dB=47, psat=50, save_plot=True):
    """
    3D Visualization of SATCOM Output Power over Effective Input Power and Attenuation.
    """
    print("Generating refined 3D plot with attenuation and effective input power...")

    # Set axis ranges
    effective_input_min = 0
    effective_input_max = df['Effective_Input_Power_dBm'].max()

    attenuation_min = df['Attenuation_D_B'].min()
    attenuation_max = df['Attenuation_D_B'].max()

    # Create grid for Effective Input Power and Attenuation
    effective_input_range = np.linspace(effective_input_min, effective_input_max, 100)
    attenuation_range = np.linspace(attenuation_min, attenuation_max, 100)

    # Create meshgrid
    X_eff_input, Y_attenuation = np.meshgrid(effective_input_range, attenuation_range)

    # Compute Output Power using the logistic compression model
    Z_output_power = logistic_compression(
        X_eff_input,
        gain=fixed_hpa_gain,
        p1dB=p1dB,
        psat=psat,
        smoothness=0.5
    )

    # Colormap
    colors = [(0.0, "green"), (0.5, "yellow"), (0.75, "orange"), (1.0, "red")]
    cmap = LinearSegmentedColormap.from_list("green_yellow_red", colors)

    # Plot 3D Surface + Actual Points
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter actual data points
    ax.scatter(
        df['Effective_Input_Power_dBm'],
        df['Attenuation_D_B'],
        df['Output_Power_dBm'],
        color='blue',
        s=50,
        alpha=0.6,
        label='Actual Output Power'
    )

    # Surface plot of the compression model prediction
    surf = ax.plot_surface(
        X_eff_input,
        Y_attenuation,
        Z_output_power,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.8
    )

    # Axis labels and limits
    ax.set_xlabel('Effective Input Power (dBm)')
    ax.set_ylabel('Attenuation (dB)')
    ax.set_zlabel('Output Power (dBm)')

    ax.set_xlim(effective_input_min, effective_input_max)
    ax.set_ylim(attenuation_min, attenuation_max)
    ax.set_zlim(0, psat + 5)

    ax.set_title(f'Smoothed SATCOM Output Power Prediction\n'
                 f'(Fixed HPA Gain = {fixed_hpa_gain} dB)\n'
                 f'Compression (P1dB={p1dB} dBm, Psat={psat} dBm)')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Output Power (dBm)')

    ax.view_init(elev=25, azim=30)

    plt.show()

    # ===============================================
    # PART 2: 2D REGRESSION PLOT + SAVE (Logistic Fit)
    # ===============================================
    if save_plot:
        print("\nCreating 2D regression plot...")

        df_filtered = df[(df['Attenuation_D_B'] <= 0.1) & (df['Output_Power_dBm'] >= 35)]
        print(f"Filtered data points: {len(df_filtered)}")

        if df_filtered.empty:
            print("⚠️  No data points found with Attenuation == 0 and Output Power >= 35 dBm.")
            return

        X_input = df_filtered['Input_Power_dBm'].values
        y_output = df_filtered['Output_Power_dBm'].values

        sorted_indices = np.argsort(X_input)
        X_sorted = X_input[sorted_indices]
        y_sorted = y_output[sorted_indices]

        # Logistic model function for curve fit
        def logistic_model(x, gain, p1dB, psat, smoothness):
            linear_output = x + gain
            return psat / (1 + np.exp(-smoothness * (linear_output - p1dB)))

        bounds_lower = [10, 40, 45, 0.01]
        bounds_upper = [50, 55, 55, 5.0]

        popt, _ = curve_fit(
            logistic_model,
            X_sorted,
            y_sorted,
            p0=[fixed_hpa_gain, p1dB, psat, 0.5],
            bounds=(bounds_lower, bounds_upper)
        )

        print(f"Fitted params: gain={popt[0]:.2f}, p1dB={popt[1]:.2f}, psat={popt[2]:.2f}, smoothness={popt[3]:.2f}")

        y_pred_curve = logistic_model(X_sorted, *popt)

        fig2 = plt.figure(figsize=(10, 6))
        plt.scatter(X_sorted, y_sorted, color='blue', alpha=0.6, label='Actual Output Power')
        plt.plot(X_sorted, y_pred_curve, color='red', linewidth=2, label='Logistic Fit')

        plt.xlabel('Input Power (dBm)')
        plt.ylabel('Output Power (dBm)')
        plt.title('SATCOM Output Power vs Input Power\n(Attenuation = 0 dB, Output >= 35 dBm)')
        plt.legend()
        plt.grid(True)

        plt.show()

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        base_filename = "SATCOM_Input_vs_Output_LogisticFit"
        extension = ".png"
        filename = os.path.join(script_dir, f"{base_filename}{extension}")
        file_index = 1

        while os.path.exists(filename):
            filename = os.path.join(script_dir, f"{base_filename}_{file_index}{extension}")
            file_index += 1

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig2)

        print(f"Filtered 2D plot saved successfully to: {filename}")


# 6. Save the Model and Preprocessing Objects
def save_model(model, scaler, poly):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    print(f"Saving models to directory: {script_dir}")

    model_path = os.path.join(script_dir, 'satcom_model.pkl')
    scaler_path = os.path.join(script_dir, 'satcom_scaler.pkl')
    poly_path = os.path.join(script_dir, 'satcom_poly.pkl')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(poly, poly_path)

    print(f"Model and preprocessors saved to:\n{script_dir}")

# 7. Main Execution Block
if __name__ == '__main__':
    p1dB = 47
    psat = 50
    fixed_hpa_gain = 30

    df = generate_satcom_data(samples=1000, fixed_hpa_gain=fixed_hpa_gain, p1dB=p1dB, psat=psat)

    X_poly, y, scaler, poly = preprocess_data(df, degree=2)

    model = train_model(X_poly, y, alpha=1.0)

    y_pred = evaluate_model(model, X_poly, y, cv_folds=5)

    visualize_model(df, model, scaler, poly, fixed_hpa_gain=fixed_hpa_gain, p1dB=p1dB, psat=psat)

    save_model(model, scaler, poly)
