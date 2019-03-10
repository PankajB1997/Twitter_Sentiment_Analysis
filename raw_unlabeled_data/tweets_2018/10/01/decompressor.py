import os
import bz2
import json
import pickle

print("Each pickle file here is a list of english tweets for the given hour of the day.\n")

folders = [ item for item in os.listdir() if not item == "decompressor.py" and not item == "result" and not item == "backup" ]

totalcount = 0

if not os.path.exists("result"):
    os.mkdir("result")

for folder in folders:
    zipfiles = [ item for item in os.listdir(folder) if item[-4:] == ".bz2" ]
    otherfiles = [ item for item in os.listdir(folder) if not item[-4:] == ".bz2" ]
    # Delete any existing non-bz2 files
    for file in otherfiles:
        if os.path.exists(os.path.join(folder, file)):
            os.remove(os.path.join(folder, file))
    tweets = []
    for zipfile in zipfiles:
        with bz2.BZ2File(os.path.join(folder, zipfile), 'r') as file:
            data = [ json.loads(item.decode("utf-8")[ item.decode("utf-8").index("{") : item.decode("utf-8").rindex("}")+1 ]) for item in file.readlines() ]
            data = [ item["text"] for item in data if "text" in item and "lang" in item and item["lang"] == "en" ]
            if len(data) > 0:
                tweets.extend(data)
                totalcount += len(data)
    print("Number of tweets added for hour " + folder + ": " + str(len(tweets)) + ", Total tweets so far: " + str(totalcount))
    pickle.dump(tweets, open(os.path.join("result", folder + ".pkl"), 'wb'))