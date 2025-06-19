import matplotlib.pyplot as plt
# Import necessary libraries
import numpy as np
import pandas as pd
import ruptures as rpt
from hmmlearn import hmm

from data_processing.fissures_processing import chargement_donnees

# Upload
fissures_path = "data/Fissures/"
df, df_old = chargement_donnees(fissures_path)

# Apply the scaling factor to the 'Bureau' column in df to match 'Bureau_old'
df["Bureau_scaled"] = df["Bureau"] * 1.12

# Rename the 'Bureau_old' column for consistency
df_old.rename(columns={"Bureau_old": "Bureau_scaled"}, inplace=True)

# Combine the two datasets
df_combined = df_old

# Extract the values for the analysis
ecartement_values = df_combined["Bureau_scaled"].values
dates = df_combined["Date"].values

# 1. Structural Break Analysis (Ruptures)
algo = rpt.Pelt(model="l2").fit(ecartement_values)
breakpoints = algo.predict(pen=10)  # Tune the 'pen' parameter as needed

# Remove the last breakpoint (it corresponds to the end of the dataset and causes out-of-bounds error)
breakpoints = breakpoints[:-1]

# Convert breakpoints to match dates
breakpoint_dates = [df_combined["Date"].iloc[bp] for bp in breakpoints]

# Plot the results for structural break detection with dates on the x-axis
plt.figure(figsize=(10, 6))
plt.plot(df_combined["Date"], ecartement_values, label="Ecartement")
for bp in breakpoint_dates:
    plt.axvline(x=bp, color="r", linestyle="--", label=f"Rupture at {bp.date()}")
plt.title("Rupture Detection with PELT Algorithm")
plt.xlabel("Date")
plt.ylabel("Ecartement (mm)")
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.legend()
plt.show()

plt.show()


# 2. Hidden Markov Model (HMM)
n = 2  # 3 ou 2 (une phase log puis une phase exp ?) pour new et 2 pour old

# Reshape the data for HMM
X = ecartement_values.reshape(-1, 1)

# Fit a Gaussian Hidden Markov Model
hmm_model = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=1000)
hmm_model.fit(X)

# Predict the hidden states
hidden_states = hmm_model.predict(X)

# Plot the hidden states with respect to time
plt.figure(figsize=(10, 6))
plt.plot(dates, ecartement_values, label="Ecartement (mm)", color="blue")
for i in range(n):
    plt.plot(
        dates[hidden_states == i],
        ecartement_values[hidden_states == i],
        "o",
        label=f"State {i}",
    )
plt.title("Hidden Markov Model - Hidden States Over Time")
plt.xlabel("Date")
plt.ylabel("Ecartement (mm)")
plt.legend()
plt.show()
