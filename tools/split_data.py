from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def draw_graph(series, title):
    series = series.sort_index(ascending=True)
    series.plot.bar()
    plt.title(title)
    plt.show()


data_path = "../data/all_data.csv"
df = pd.read_csv(data_path)
df = df.sample(frac=1)

all_age_graph = df.value_counts('age')
all_gender_graph = df.value_counts('gender')
all_race_graph = df.value_counts('race')

# draw_graph(all_age_graph, "Age Distribution")
# draw_graph(all_gender_graph, "Gender Distribution")
# draw_graph(all_race_graph, "Race Distribution")

TRAIN_SPLIT = int(0.8 * len(df))
VAL_SPLIT = int(0.1 * (len(df) - TRAIN_SPLIT))
TEST_SPLIT = int(len(df) - (TRAIN_SPLIT + VAL_SPLIT))

TRAIN_DATA = df[0:TRAIN_SPLIT]
VAL_DATA = df[TRAIN_SPLIT: TRAIN_SPLIT + VAL_SPLIT]
TEST_DATA = df[TRAIN_SPLIT + VAL_SPLIT:]

TRAIN_DATA.to_csv("../data/train.csv", index=None)
VAL_DATA.to_csv("../data/val.csv", index=None)
TEST_DATA.to_csv("../data/test.csv", index=None)

# print(TRAIN_DATA.value_counts("gender"))
