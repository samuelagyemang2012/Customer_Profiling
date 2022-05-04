import pandas as pd
import os

images_path = "C:/Users/Administrator/Desktop/datasets/UTKface_Aligned_cropped/UTKFace/UTKFace/"
data_path = "../data/"
images = os.listdir(images_path)
data = []

"""
Data format
[age]_[gender]_[race]_[date&time]
age -> 0-116
gender -> 0(male) or 1(female)
race -> White(0), Black(1), Asian(2), Indian(3), Others(4) (like Hispanic, Latino, Middle Eastern).
"""

for image in images:
    file = image.split("_")
    age = file[0]
    gender = file[1]
    race = file[2]
    data.append([image, age, gender, race])

df = pd.DataFrame(data, columns=["names", "age", "gender", "race"])
df.to_csv(data_path + "all_data.csv", index=None)
