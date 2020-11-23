import pandas as pd
import spacy

data = pd.read_csv("Sentiment_Analysis_Cleaned_2.csv", encoding="ISO-8859-1")

nlp = spacy.load("/Users/tanushnadimpalli/Documents/python_stuff_new")

texts = ["this show is really good",
         "I love this movie!",
         "I hate this movie!",
         "I’ve been to the year 3000 not much has changed, except everyone was wearing a mask.",
         "holy yes. excited to vote!! still amazed Election Day isn't a national holiday. it needs to be celebrated, with a post vote bar crawl.",
         "how far down did you have to scroll on the timeline to see this tweet? Like was it at the top chronologically or did you have to like SCROLL",
         "I am 38 years old and still terrified every time I try to cut a bagel in half.",
         ]
docs = [nlp.tokenizer(text) for text in texts]

# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

print(scores)
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])

predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])


