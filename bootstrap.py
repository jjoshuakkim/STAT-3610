import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Define the dataset using our sample information
data = np.array([12, 9.4, 10, 13.5, 9.3, 10.1, 9.6, 9.3, 9.1, 9.2, 11, 9.1, 10.4, 9.1, 13.3, 10.6])

#%%
# Define the number of resamples
n_resamples = 200
# Size of resamples
n = 16

# Create an array to store the resamples
resamples = np.empty(n_resamples)

# Computes bootstrapped values of S2
for i in range(n_resamples):
    # Resample the data with replacement
    temp = np.random.choice(data, size=n, replace=True)

    # Calculates the sample var for each resample
    resamples[i] = np.var(temp)

# Builds histogram
plt.hist(resamples, bins=20)
plt.title("Bootstrapped Values of S2")
plt.xlabel("S2")
plt.show()

# Create an array to store the random samples
random_samples = np.empty(n_resamples)
theta = np.mean(data) - 9

# Perform bootstrapping
for i in range(n_resamples):
    # Generates a single random sample from the ex dist
    rs = np.random.exponential(scale=1/theta, size=n)

    # Calculates the sample var for each random sample
    random_samples[i] = np.var(rs)

# Build histogram
plt.hist(random_samples, bins=20)
plt.title("Simulated Values of S2")
plt.xlabel("S2")
plt.show()

# Generate QQ plot of two sets of sample variances
sm.qqplot_2samples(resamples, random_samples)
plt.title("Q-Q Plot of Sample Variances")
plt.show()
