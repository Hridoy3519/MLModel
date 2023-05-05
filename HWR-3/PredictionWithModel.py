import os
import HWR


def predict():
    lines = []
    with open("cropped_images/img-sequence.txt", 'r') as file:
        lines = file.read().splitlines()
    print(lines)
    x = HWR.predict_by_saved_model(lines)
    new = ""
    for item in x:
        for key, value in item.items():
            new += " ".join(value) + " "
    return new.strip()

#
# x=predict()
# print(x)
