import zmq
import json
import os
import logging
from datetime import datetime

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")

logging.basicConfig(filename='03_Data/Logs/ml_server.log', level=logging.INFO,
                    format='%(asctime)s | %(message)s')

print("ML_SERVER LIVE â€"" Listening on 5555")
while True:
    msg = socket.recv_json()
    with open("03_Data/latest_features.json", "w") as f:
        json.dump(msg, f)
    logging.info(f"FEATURES RECEIVED | {len(msg['features'])} | {msg['symbol']}")





