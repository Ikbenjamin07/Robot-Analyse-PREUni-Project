from Julia_single import SF
from anthro_single import anthro
from angeles_single import angle 
from circles_single import circle

def main():
    spatial_frequency  = SF()
    print(spatial_frequency[0])
    anthro_score = anthro()
    print(anthro_score)
    circles = circle()
    print(circles)
    angle_score = angle()
    print(angle_score)
    predicted_offer = 0.7182 - 0.0806*spatial_frequency[0] + 0.0437 * angle_score[0] + 0.0241 * anthro_score[0] + 0.0120 * circles[0]
    print(predicted_offer)


if __name__ == "__main__":
    main()