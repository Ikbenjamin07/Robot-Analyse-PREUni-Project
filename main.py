from Julia import SF
from dictator_offer import Get_Lists
import pandas as pd

def main():
    robots, dictator_offer = Get_Lists()
    spatial_frequency  = SF()

    df = pd.DataFrame({'names': robots, 'offers': dictator_offer, 'SF': spatial_frequency})
    print(df)

if __name__ == "__main__":
    main()