#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:22:13 2018

@author: lappv
"""

from flask import Flask, request, jsonify
from keras.models import load_model
import loss
import os
import ForHomeber as AICore
import tensorflow as tf

app = Flask(__name__)

def init():
    app.config.from_object('config')
    loadModel()
    
    
def loadModel():
    baseDir = os.path.abspath(os.path.dirname(__file__))
    #load the pre-trained Keras model, set default graph
    global houseModel, graph
    graph = tf.get_default_graph()
    houseModel = load_model(baseDir + "/model_house_ures_selu.hdf5",
                                      custom_objects={
                                             'bce_dice_loss': loss.bce_dice_loss,
                                             'binary_accuracy': loss.binary_accuracy,
                                     })
    
# API get garden border
@app.route("/garden/border", methods=["GET"])
def getGardenBorder():
    photoType = request.args.get('photo_type')
    with graph.as_default():
        if (photoType == 'url'):
            (coordinate, contours) = AICore.predict_latlong_house_url(request.args.get('photo'), app.config['WIDTH'], app.config['HEIGHT'], parseCoordinate(request.args.get('coordinate')), houseModel)
        else:
            (coordinate, contours) = AICore.predict_latlong_house_path(request.args.get('photo'), app.config['WIDTH'], app.config['HEIGHT'], parseCoordinate(request.args.get('coordinate')), houseModel)
    print(coordinate)
    corners = parseCoordinate(request.args.get('coordinate'))
    response = toBorderResponse(coordinate)
    response.append({'lat':corners[0][0], 'lon':corners[0][1]})
    response.append({'lat':corners[1][0], 'lon':corners[1][1]})
    response.append({'lat':corners[2][0], 'lon':corners[2][1]})
    response.append({'lat':corners[3][0], 'lon':corners[3][1]})
    return jsonify({'border': response})

    
def parseCoordinate(coordinateJson):
    coordinates = coordinateJson.split(',')
    return [
                [float(coordinates[0]), float(coordinates[1])],
                [float(coordinates[2]), float(coordinates[3])],
                [float(coordinates[4]), float(coordinates[5])],
                [float(coordinates[6]), float(coordinates[7])],
           ]

def toBorderResponse(aiResponse):
    border = []
    for point in aiResponse[0].tolist():
        border.append({'lat': point[0], 'lon': point[1]})
    
    return border

if __name__ == "__main__":
    init()
    app.run(debug=True, host='0.0.0.0')
