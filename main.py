from Julia import SF
from dictator_offer import Get_Lists
from anthro_all import anthro
import pandas as pd

def main():
    robots, dictator_offer = Get_Lists()
    spatial_frequency  = SF()
    anthro_score = anthro()

    df = pd.DataFrame({'names': robots, 'offers': dictator_offer, 'SF': spatial_frequency, 'anthro': anthro_score})
    print(df)

if __name__ == "__main__":
    main()