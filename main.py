from Julia import SF
from dictator_offer import Get_Lists
from anthro_all import anthro
import pandas as pd
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from angeles_3_0 import angle 
from circles_2_0 import circle


def main():
    #robots, dictator_offer = Get_Lists()
    #spatial_frequency  = SF()
    #anthro_score = anthro()
    #circles = circle()
    #angle_score = angle()
    #df = pd.DataFrame({'names': robots, 'offers': dictator_offer, 'SF': spatial_frequency, 'anthro': anthro_score, 'angle': angle_score, 'circle': circles})
    ##print(df)
    #df.to_csv('df.csv')

    df = pd.read_csv('df.csv')


    X = df[['SF', 'angle', 'anthro', 'circle']]
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
    #ax.set_xlabel("Predicted dictator offer (optimal)")
    #ax.set_ylabel("Observed dictator offer")
    #X_ref = Y_ref = np.linspace(Y_min, Y_max, 100)
    #plt.plot(X_ref, Y_ref, color='red', linewidth=1)
    ##print(X_ref)
    ##print(Y_ref)
    #plt.show()

    #print(r_squared)
    #print(r_value) 

    print(results.summary())

if __name__ == "__main__":
    main()