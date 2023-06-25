# Facial Recognition Assignment MS548
# Kenneth A Carr

import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

running = True
run_facial_recognition = True
while(running):
    image_file_path = input("Enter Image File Path To Anylyise: or E - EXIT: ")
    image = cv2.imread(image_file_path)
    if(image_file_path.lower() == "e"):
        running = False
        run_facial_recognition = False
    if(run_facial_recognition == True):
        plt.imshow(image[:, :, : : -1])
        plt.title("*** Close Window For Facial Scan Results Of Image***\n")
        plt.show()
        age_results = DeepFace.analyze(img_path = image_file_path,
                                       actions = ['age', 'emotion'])
        print("Age: ")
        print(age_results[0]['age'])

        print("Emotion: ")
        print(age_results[0]['dominant_emotion'])

    if(run_facial_recognition == False):
        print("GOODBYE!")