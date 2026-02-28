import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# -----------------------------
# 1. Load Excel file
# -----------------------------
file_path = r"C:\GSR_data.xlsx"  # Update this to your file path
df = pd.read_excel(file_path)

# Verify required columns
required_cols = ['Subject_ID', 'GSR_IC', 'GSR_MODULE']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"Excel file must contain columns: {required_cols}")

subjects = df['Subject_ID'].unique()

# -----------------------------
# 2. Create folder for results
# -----------------------------
results_folder = r"C:\GSR_Validation_Results"
os.makedirs(results_folder, exist_ok=True)

# -----------------------------
# 3. Direct comparison
# -----------------------------
mae_direct = mean_absolute_error(df['GSR_IC'], df['GSR_MODULE'])
rmse_direct = np.sqrt(mean_squared_error(df['GSR_IC'], df['GSR_MODULE']))
r2_direct = r2_score(df['GSR_IC'], df['GSR_MODULE'])

print("=== Direct Comparison ===")
print(f"MAE  : {mae_direct:.2f}")
print(f"RMSE : {rmse_direct:.2f}")
print(f"R²   : {r2_direct:.3f}")

# -----------------------------
# 4. Linear regression equivalence
# -----------------------------
X = df['GSR_IC'].values.reshape(-1, 1)
y = df['GSR_MODULE'].values
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

slope = reg.coef_[0]
intercept = reg.intercept_
r2_reg = r2_score(y, y_pred)

print("\n=== Linear Regression Equivalence ===")
print(f"GSR_MODULE = {slope:.4f} * GSR_IC + {intercept:.4f}")
print(f"R² = {r2_reg:.3f}")

# Plot Regression
plt.figure(figsize=(6,6), dpi=300)
plt.scatter(df['GSR_IC'], df['GSR_MODULE'], color='blue', label='Data points')
plt.plot(df['GSR_IC'], y_pred, color='red', linewidth=2, label='Regression line')
plt.plot(df['GSR_IC'], df['GSR_IC'], '--', color='black', label='Identity line (y=x)')
plt.xlabel('GSR_IC')
plt.ylabel('GSR_MODULE')
plt.title('Linear Regression Equivalence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "Regression_Equivalence.png"), dpi=300)
plt.show()

# -----------------------------
# 5. ML-Based LOSO Validation (SVR)
# -----------------------------
svr_pred_all = []
ic_all = []

for test_subject in subjects:
    train_data = df[df['Subject_ID'] != test_subject]
    test_data = df[df['Subject_ID'] == test_subject]

    X_train = train_data[['GSR_MODULE']].values
    y_train = train_data['GSR_IC'].values
    X_test = test_data[['GSR_MODULE']].values
    y_test = test_data['GSR_IC'].values

    # SVR (linear)
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)

    svr_pred_all.extend(y_pred_svr)
    ic_all.extend(y_test)

# Convert to numpy arrays
svr_pred_all = np.array(svr_pred_all)
ic_all = np.array(ic_all)

# Compute overall metrics
mae_svr = mean_absolute_error(ic_all, svr_pred_all)
rmse_svr = np.sqrt(mean_squared_error(ic_all, svr_pred_all))
r2_svr = r2_score(ic_all, svr_pred_all) if len(ic_all) > 1 else np.nan

print("\n=== ML-Based LOSO Validation (SVR) ===")
print(f"MAE  : {mae_svr:.2f}")
print(f"RMSE : {rmse_svr:.2f}")
print(f"R²   : {r2_svr if not np.isnan(r2_svr) else 'undefined (single measurement per subject)'}")

# -----------------------------
# 6. Bland-Altman Plot (SVR)
# -----------------------------
mean_val = (ic_all + svr_pred_all) / 2
diff_val = svr_pred_all - ic_all
bias = np.mean(diff_val)
loa_upper = bias + 1.96 * np.std(diff_val)
loa_lower = bias - 1.96 * np.std(diff_val)

plt.figure(figsize=(6,6), dpi=300)
plt.scatter(mean_val, diff_val, color='blue')
plt.axhline(bias, color='red', linestyle='--', label=f'Bias={bias:.2f}')
plt.axhline(loa_upper, color='green', linestyle='--', label=f'+1.96 SD={loa_upper:.2f}')
plt.axhline(loa_lower, color='green', linestyle='--', label=f'-1.96 SD={loa_lower:.2f}')
plt.xlabel('Mean GSR')
plt.ylabel('Difference (SVR - IC)')
plt.title('Bland-Altman Plot (SVR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "BlandAltman_SVR.png"), dpi=300)
plt.show()

print(f"\nBland-Altman Bias: {bias:.2f}")
print(f"Limits of Agreement: [{loa_lower:.2f}, {loa_upper:.2f}]")
