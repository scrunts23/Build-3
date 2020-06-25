import sys
import os
import requests
import flask
from flask import Flask, request, jsonify, Blueprint

import pickle
import sklearn
import pandas

recommend_route = Blueprint('recommend_route',__name__)

@recommend_route.route('/predict', methods=['GET','POST'])


def predict():
    '''
    Returns a JSON object of the recommended strain information 
    calculated from a users flavor and effect. 
    '''
    
    # Load Model
    KNN_FILEPATH = 'web_app/data/knn_rd.pkl'
    print("LOADING THE MODEL...")
    with open(KNN_FILEPATH, "rb") as model_file:
        saved_model = pickle.load(model_file)
    
    # Load Effects
    EFFECTS_FILEPATH = 'web_app/data/effects.pkl'
    print("LOADING EFFECTS..")
    with open(EFFECTS_FILEPATH, "rb") as effect_file:
        effects = pickle.load(effect_file)
    
    # Load Flavors
    FLAVOR_FILEPATH = 'web_app/data/flavors.pkl'
    print("LOADING FLAVORS...")
    with open(FLAVOR_FILEPATH, "rb") as flavors_file:
        flavors = pickle.load(flavors_file)
    
    from_web = request.get_json() or request.args
    effe = from_web['effect']
    flav = from_web['flavor']

    
    # Generate Recommendation
    print("RECOMMENDING...")
    

    # Use info to get vectors from pickled dictionaries
    effect = effects[effe]
    flavor = flavors[flav]

    # Generate query vector by adding these vectors
    query = effect + flavor

    # Run knn model 
    result = saved_model.kneighbors(query.reshape(1,-1))
    DF_FILEPATH = 'data_wrangling/raw_csv/cannabis.csv'
    df = pandas.read_csv(DF_FILEPATH)

    # Result object will have the index location of recomendations to lookup in df
    strains = df.iloc[result[1][0]]['Strain'].to_list() 
    recommendation_dictionaries = []
    for i in range(2):
        rec = df[df['Strain']== strains[i]].reset_index()
        rec.columns =  ['id', 'strain', 'type','rating', 'effect', 'flavor', 'description']
        dictionary = rec.to_dict()
        recommendation_dictionaries.append(dictionary)
        
        
    return jsonify(recommendation_dictionaries ) 