import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.formula.api as smf

df = pd.read_csv("spx_pereturns_blog1.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
df['close'] = pd.to_numeric(df['close'], errors='coerce')

for col in ['1yr','5yr','10yr']:
    df[col] = df[col].astype(str).str.replace('%','').astype(float)

df.dropna(subset=['pe','1yr','5yr','10yr'], inplace=True)

model_1y = smf.ols("Q('1yr') ~ pe", data=df).fit()
model_5y = smf.ols("Q('5yr') ~ pe", data=df).fit()
model_10y = smf.ols("Q('10yr') ~ pe", data=df).fit()

print(model_1y.summary())
print(model_5y.summary())
print(model_10y.summary())

sns.set_theme(style='white', context='talk', rc={'axes.spines.top': False,'axes.spines.right': False})

def plot_regression(xcol, ycol, title):
    plt.figure(figsize=(8,6))
    ax = sns.regplot(
        x=xcol, 
        y=ycol, 
        data=df,
        scatter_kws={'alpha':0.8, 'marker':'D', 'facecolors':'none', 'edgecolors':'black'},
        line_kws={'color':'red'},
        ci=95
    )
    plt.axvline(x=30, color='grey', linestyle='--', lw=1.5)
    plt.axhline(y=0, color='grey', linestyle='--', lw=1.2)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
    plt.xlabel('P/E Ratio')
    plt.ylabel('Forward Return (%)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_regression('pe','1yr','SPX 1-Year Forward Return vs. P/E')
plot_regression('pe','5yr','SPX 5-Year Forward Return vs. P/E')
plot_regression('pe','10yr','SPX 10-Year Forward Return vs. P/E')
