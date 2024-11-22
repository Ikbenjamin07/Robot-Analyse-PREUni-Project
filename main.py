from Julia import SF
from dictator_offer import Get_Lists
from anthro_all import anthro
import pandas as pd
from scipy import stats
from sklearn import linear_model
from find_circels import circle
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from angeles_2_0 import angle 


def main():
    robots, dictator_offer = Get_Lists()
    spatial_frequency  = SF()
    anthro_score = anthro()
    #circles = circle()
    angle_score = angle()
    df = pd.DataFrame({'names': robots, 'offers': dictator_offer, 'SF': spatial_frequency, 'anthro': anthro_score, 'angle': angle_score})
    #print(df)
    df.to_csv('df.csv')

    #df = pd.read_csv('df.csv')
    res = stats.pearsonr(df['angle'], df['offers'])
    print(res)

    X = df[['SF', 'anthro', 'angle']]
    y = df['offers']

    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    #print(regr.coef_)

    X = sm.add_constant(X)
    model = sm.OLS(y,X)
    results = model.fit()
    
    Y_max = y.max()
    Y_min = y.min()

    #ax = sns.scatterplot(x=results.fittedvalues, y=y)
    #ax.set(ylim=(Y_min, Y_max))
    #ax.set(xlim=(Y_min, Y_max))
    #ax.set_xlabel("predicted offer")
    #ax.set_ylabel("observed offer")
    #X_ref = Y_ref = np.linspace(Y_min, Y_max, 100)
    #plt.plot(X_ref, Y_ref, color='red', linewidth=1)
    ##print(X_ref)
    ##print(Y_ref)
    #plt.show()

    r_squared = results.rsquared
    r_value = r_squared**0.5
    #print(r_squared)
    #print(r_value) 

    print(results.summary())

if __name__ == "__main__":
    main()