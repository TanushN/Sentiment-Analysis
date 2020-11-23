import spacy
import pandas as pd
import spacy
from spacy.util import minibatch,compounding
import random
from spacy.lang.en.stop_words import STOP_WORDS

data = pd.read_csv("Sentiment_Analysis_Cleaned_2.csv", encoding="ISO-8859-1")

print(data.head())


model_directory = "/Users/tanushnadimpalli/Documents/python_stuff_new"

if model_directory:
    print("loading model from disk")
    nlp = spacy.load(model_directory)
else:
    print("making new blank model")
    # create a empty model
    nlp = spacy.blank("en")

    # Create the TextCategorizer with exclusive classes and "bow" architecture
    textcat = nlp.create_pipe(
        "textcat",
        config={
            "exclusive_classes": True,
            "architecture": "ensemble"})

    # Add the TextCategorizer to the empty model
    nlp.add_pipe(textcat)

    textcat.add_label("Negative")
    textcat.add_label("Positive")

train_texts = data['tweet'].values
train_labels = [{'cats': {'Negative': label == 'Negative',
                          'Positive': label == 'Positive'}}
                for label in data['target']]

train_data = list(zip(train_texts, train_labels))

print(train_data[:3])

pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
with nlp.disable_pipes(*other_pipes):  # only train textcat
    random.seed(1)
    spacy.util.fix_random_seed(1)
    optimizer = nlp.begin_training()
    losses = {}
    batch_sizes = 10000
    for epoch in range(10):
        print('starting epoch')
        random.shuffle(train_data)
        # Create the batch generator with batch size = 8
        batches = minibatch(train_data, size=batch_sizes)
        # Iterate through minibatches
        for batch in batches:
            # Each batch is a list of (text, label) but we need to
            # send separate lists for texts and labels to update().
            # This is a quick way to split a list of tuples into lists
            texts, labels = zip(*batch)
            nlp.update(texts, labels, sgd=optimizer, drop=0.2, losses=losses)

        print(losses)

        with nlp.use_params(optimizer.averages):
            nlp.to_disk("/Users/tanushnadimpalli/Documents/python_stuff_new")
        print("Saved model")

texts = ["My friend hates twitter as he finds it very annoying", "I love your new shoes, they look very stylish"]
docs = [nlp.tokenizer(text) for text in texts]

# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

print(scores)

predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])

# with nlp.use_params(optimizer.averages):
#     nlp.to_disk("/Users/tanushnadimpalli/Documents/python_stuff_new")
# print("Saved model")

