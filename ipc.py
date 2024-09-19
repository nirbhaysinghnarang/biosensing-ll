import zmq
import time
import json
ctx = zmq.Context()
# The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
pupil_remote = ctx.socket(zmq.REQ)

ip = '127.0.0.1'  # If you talk to a different machine use its IP.
port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

pupil_remote.connect(f'tcp://{ip}:{port}')

# Request 'SUB_PORT' for reading data
pupil_remote.send_string('SUB_PORT')
sub_port = pupil_remote.recv_string()

# Request 'PUB_PORT' for writing data
pupil_remote.send_string('PUB_PORT')
pub_port = pupil_remote.recv_string()


subscriber = ctx.socket(zmq.SUB)
subscriber.connect(f'tcp://{ip}:{sub_port}')
subscriber.subscribe('gaze.')  # receive all gaze messages

# we need a serializer
import msgpack

def extract_diameters(gaze_data):
    base_data = gaze_data['base_data'][0]
    diameter = base_data['diameter']
    diameter_3d = base_data['diameter_3d']
    return diameter, diameter_3d


while True:
    topic, payload = subscriber.recv_multipart()
    message = msgpack.loads(payload)
    diameter, diameter_3d = extract_diameters(message)
    
    print(diameter, diameter_3d)
    time.sleep(5)

    
    
