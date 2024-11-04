from Julia import SF
from dictator_offer import Get_Lists
from anthro_all import anthro
import pandas as pd
from scipy import stats
from sklearn import linear_model
from find_circels import circle
import statsmodels.api as sm


def main():
    robots, dictator_offer = Get_Lists()
    spatial_frequency  = SF()
    anthro_score = anthro()
    circles = circle()

    df = pd.DataFrame({'names': robots, 'offers': dictator_offer, 'SF': spatial_frequency, 'anthro': anthro_score, 'circles': circles})
    print(df)
    df.to_csv('df.csv')
    #df = pd.read_csv('df.csv')
    res = stats.pearsonr(df['circles'], df['offers'])
    print(res)

    X = df[['anthro', 'SF', 'circles']]
    y = df['offers']

    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    print(regr.coef_)

    X = sm.add_constant(X)
    model = sm.OLS(y,X)
    results = model.fit()
    print(results.params)

    r_squared = results.rsquared
    r_value = r_squared**0.5

    print(r_squared)
    print(r_value)

    print(results.summary())


if __name__ == "__main__":
    main()