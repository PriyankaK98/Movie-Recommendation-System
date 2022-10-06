import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
import tflearn
import heapq
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import Dense
from keras.models import Sequential
from flask import Flask, request, jsonify, render_template, Response
import json


stemmer = LancasterStemmer();

app = Flask('__name__')

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return act[act.index == index]["title"].values[0]

def get_features_from_title(title):
	return act[act.title == title]["combined_features"].values[0]
    #return df[df.title == title]["combined_features"]

##################################################



##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")
act = df.copy()
# #print(act['genres'].head(10))
act['title'] = act['title'].str.lower()
act['genres'] = act['genres'].str.lower()
act['cast'] = act['cast'].str.lower()
# print(act['title'].head(10))
# print(df['title'].head(10))
#print df.columns
##Step 2: Select Features

features = ['keywords','cast','genres','director']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
	act[feature] = act[feature].fillna('')
words = []
labels = []
docs_x = []
docs_y = []
commons = []


def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print("Error:", row)

act["combined_features"] = act.apply(combine_features,axis=1)

#print("Combined Features:", df["combined_features"].head())

for patterns in act["combined_features"]:
    wrds = nltk.word_tokenize(patterns)
    words.extend(wrds)
    docs_x.append(wrds)

for common in act:
    docs_y.extend(act['combined_features'])
    labels.extend(act['combined_features'])
    commons.extend(act['combined_features'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))        #removes duplicate words from the list
labels = sorted(list(set(labels)))
commons = sorted(list(set(commons)))
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)
training = np.array(training)
output = np.array(output)

tf.reset_default_graph()
#print(len(training[0])," ", len(output[0]))

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #activation fumction gives the probability of each neuron in the layer.
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
#     model.load('./recommend.tflearn')
# except:
model.fit(training, output, n_epoch=150, batch_size=100, show_metric=True)
model.save('recommend.tflearn')

def bag_of_words(s, words):

    specs = get_features_from_title(s)
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(specs)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)
class Recommend():
    def predict(inp):

        #print("Start the recommendation (press 'quit' to exit)")
        # if df['title'].str.contains(inp):

        results = model.predict([bag_of_words(inp, words)])[0]

        #print(final)
        return results
    def genre_predict(movie):
        df_new = df[act['genres'].str.contains(movie,case=False)]
        df_new = df_new.sort_values('popularity', ascending=False)
        print(df_new['title'].head(10))
        genre_movies = []
        genre_movies.extend(df_new['title'].tolist())
        return genre_movies[:10]
    # def cast_predict(movie):
    #     df_cast = df[act['cast'].str.contains(movie, case=False)]
    #     df_cast = df_cast.sort_values('popularity', ascending=False)
    #     print(df_cast['title'].head(10))
    #     cast_movies = []
    #     cast_movies.extend(df_cast['title'].tolist())
    #     return cast_movies[:10]


Movie_titles = df['title'].tolist()


@app.route('/')
def home():

    return render_template('index.html')

@app.route('/autocomplete',methods=['GET'])
def autocomplete():
    search = request.args.get('term')

    app.logger.debug(search)
    return Response(json.dumps(Movie_titles), mimetype='application/json')

# @app.route('/auto',methods=['GET'])
# def auto():
#     search = request.args.get('term')
#
#     app.logger.debug(search)
#     return Response(json.dumps(Cast_titles), mimetype='application/json')


@app.route('/movieNames')
def MNames():
    return render_template('MovieName.html')

@app.route('/genreNames')
def Category():
    return render_template('genre.html')

@app.route('/Details/<string:name>',methods=['GET','POST'])
def Details(name):
    someCast= df[df.title == name]["cast"].values[0]
    director =df[df.title == name]["director"].values[0]
    genre = df[df.title == name]["genres"].values[0]
    urls = df[df.title == name]["homepage"].values[0]
    return render_template('Details.html',cast_text = someCast,movie_name = name, director= director, genre= genre,links=urls)


@app.route('/predict',methods=['POST'])
def movies():
    final = []
    infoAll=[]
    '''
    For rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    #features = str(request.form.values())
    #final_features = [np.array(features)]
    #output = Recommend.predict(features)
    features[0] = features[0].lower()
    if act['title'].isin(features).any():
        prediction = Recommend.predict(features[0])
        max = heapq.nlargest(len(prediction), range(len(prediction)), prediction.take)
        final.clear()
        infoAll.clear()
        for i in max[:11]:
            tag = commons[i]
        # print(tag)
            responses = df[act.combined_features == tag]["title"]

            final.extend(responses)
            # info = df[df.combined_features == tag]["release_date"]
            # infoAll.extend(info)
            # print(infoAll)

        return render_template('recommend.html', movie_name=features[0], scroll = 'something', prediction_text=final)

    elif act['genres'].str.contains(features[0]).any():
        g_movies = Recommend.genre_predict(features[0])

        return render_template('recommend.html', movie_name=features[0], scroll = 'something', prediction_text=g_movies)
    # elif act['cast'].str.contains(features[0]).any():
    #     c_movies = Recommend.cast_predict(features[0])
    #     return render_template('recommend.html', movie_name=features[0], prediction_text=c_movies)

    else:
        fault = ["Enter Proper Title/Genre"]
        return render_template('recommend.html', movie_name=features[0], scroll = 'something', prediction_text=fault)

    # return render_template('Details.html')





    #return render_template('index.html', len=len(final), final=final)

if __name__ == "__main__":
    app.run(debug=True)

while True:
    inp = input("Enter your favourite movie: ")
    if inp.lower() == 'quit':
        break
    final = Recommend.predict(inp)
    print(final)

