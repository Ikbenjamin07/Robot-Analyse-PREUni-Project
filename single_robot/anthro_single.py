from face_single import face
from body_shape_single import human_likeness_score

def main():
    anthro()

def anthro():
    anthro_score = []
    face_scores = face()
    body = human_likeness_score()
    for i, face_score in enumerate(face_scores):
        anthro_score.append((face_score * 10 + body[i] ))
    return anthro_score

if __name__ == "__main__":
    main()