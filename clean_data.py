import pandas as pd
import re

data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None,)
data.columns = ["target", "id", "date", "flag", "user", "tweet"]

pd.set_option('display.max_columns', None)

data = data[["target", "tweet"]]

# function to remove twitter mentions,
def clean(row):
    row = re.sub("http\S+|www.\S+", '', row)
    row = re.sub("@[A-Za-z0-9]+", "", row)
    return row

data['tweet'] = data['tweet'].apply(clean)

print(data.head())

# function to replace numbers with corresponding labels such as "Negative" and "Positive"
def map_to_values(row):
    if row == 0:
        row = "Negative"
    if row == 2:
        row = "Neutral"
    if row == 4:
        row = "Positive"
    return row

data["target"] = data["target"].apply(map_to_values)



data.to_csv('/Users/tanushnadimpalli/Documents/python_stuff/Sentiment_Analysis_Cleaned.csv', index=False, header=True)
