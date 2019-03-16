import os
import pickle
import pandas as pd

PATH = os.path.join("raw_unlabeled_data", "tweets_2018", "10")

PKL_PATHS = [ os.path.join(PATH, folder, "result") for folder in os.listdir(PATH) ]

CSV_PATHS = [ os.path.join("raw_unlabeled_data", filepath) for filepath in os.listdir("raw_unlabeled_data") if filepath[:-4] == ".csv" ]

CSV_SAVEPATH = "tweets_large_32M.csv"

size = 0

with open(CSV_SAVEPATH, "a") as csvfile:

    for path in PKL_PATHS:
        for pickle_filepath in os.listdir(path):
            tweets_current_pkl = pickle.load(open(os.path.join(path, pickle_filepath), "rb"))
            pd.DataFrame(tweets_current_pkl).to_csv(csvfile, sep=",", header=False, index=False, encoding="utf-8")
            size += len(tweets_current_pkl)

    for path in CSV_PATHS:
        tweets_current_csv = pd.read_csv(path, header=None, index_col=None)
        tweets_current_csv.to_csv(csvfile, sep=",", header=False, index=False, encoding="utf-8")
        size += tweets_current_csv.shape[0]

print("Total tweets added: ", size)
