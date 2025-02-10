import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("FEDFUNDS.csv", parse_dates=["observation_date"])
df = df[df["observation_date"] >= "1990-01-01"].copy()
df.dropna(subset=["FEDFUNDS"], inplace=True)
r = df["FEDFUNDS"].values
dt = 1/12
theta_est = np.mean(r)
dr = r[1:] - r[:-1]
sigma_est = np.std(dr, ddof=1)
X = r[:-1] - theta_est
X = sm.add_constant(X)
model = sm.OLS(dr, X).fit()
beta_hat = model.params[1]
kappa_est = -beta_hat / dt

def B(T, k):
    return (1 - np.exp(-k*T)) / k

def A(T, k, th, s):
    BT = B(T, k)
    return np.exp((th - (s**2)/(2*k**2))*(BT - T) - (s**2/(4*k))*(BT**2))

maturities = np.linspace(0.1, 30, 100)
r0 = r[-1]
vasicek_yields = []
for T in maturities:
    bp = A(T, kappa_est, theta_est, sigma_est) * np.exp(-B(T, kappa_est)*r0)
    vasicek_yields.append(-np.log(bp)/T)

plt.figure(figsize=(7, 4))
plt.plot(maturities, vasicek_yields, label="Deterministic Curve")
plt.xlabel('TTM (years)')
plt.ylabel('YTM (%)')
plt.title('Vasicek Yield Curve')
plt.legend()
plt.show()

num_paths = 200
years = 5
steps = int(years / dt)
time_grid = np.linspace(0, years, steps + 1)
paths = np.zeros((num_paths, steps + 1))
paths[:, 0] = r0
for i in range(steps):
    paths[:, i+1] = (paths[:, i]
                     + kappa_est*(theta_est - paths[:, i])*dt
                     + sigma_est*np.sqrt(dt)*np.random.randn(num_paths))

plt.figure(figsize=(7, 4))
low = np.percentile(paths, 5, axis=0)
med = np.percentile(paths, 50, axis=0)
high = np.percentile(paths, 95, axis=0)
plt.fill_between(time_grid, low, high, color="lightblue", alpha=0.5, label="5%-95% Range")
plt.plot(time_grid, med, color="blue", label="Median")
plt.title("Short-Rate Paths")
plt.xlabel("Time (years)")
plt.ylabel("Short Rate")
plt.legend()
plt.show()

final_rates = paths[:, -1]
plt.figure(figsize=(7, 4))
sns.histplot(final_rates, kde=True, color="skyblue")
plt.title("Distribution of Final Short Rates in 5 Years")
plt.xlabel("Short Rate")
plt.ylabel("Frequency")
plt.show()

maturities_future = np.linspace(0.1, 10, 50)
future_curves = []
for j in range(num_paths):
    rT = paths[j, -1]
    this_curve = []
    for m in maturities_future:
        bp = A(m, kappa_est, theta_est, sigma_est) * np.exp(-B(m, kappa_est)*rT)
        this_curve.append(-np.log(bp)/m)
    future_curves.append(this_curve)

plt.figure(figsize=(7, 4))
future_curves = np.array(future_curves)
low_c = np.percentile(future_curves, 5, axis=0)
med_c = np.percentile(future_curves, 50, axis=0)
high_c = np.percentile(future_curves, 95, axis=0)
plt.fill_between(maturities_future, low_c, high_c, color="lightgreen", alpha=0.5, label="5%-95% Range")
plt.plot(maturities_future, med_c, color="green", label="Median")
plt.title("Yield Curves in 5 Years")
plt.xlabel("Maturity (years)")
plt.ylabel("Yield")
plt.legend()
plt.show()

print("theta:", theta_est)
print("sigma (monthly):", sigma_est)
print("kappa:", kappa_est)
print(model.summary())
