from scipy import signal

import paho.mqtt.client as mqtt
import numpy as np
import pickle
from datetime import datetime as dt
import keras
import tensorflow as tf
from keras.models import load_model
import json
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

# init
indices = (0, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27)
input_data = []
tip_data = []
tip_ts = []
vis_data = []
vis_time = []
t0 = None
pred = []

vis_moved = None
tip_data = []
hips_data = []
tip_ts = []

t0 = None

max_buffer_size = 50
visual_buffer = []
sensor_buffer = []


# ------------------------------ Model ------------------------------ #
class MPJE(keras.metrics.Metric):
    def __init__(self):
        super().__init__(name='mean_per_joint_error')
        self.mpje = self.add_weight(name='mpje', dtype="float32", initializer="zeros")
        self.num_samples = self.add_weight(name='num_samples', dtype="float32", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape predictions and ground truth to [batch_size, num_joints, 3]
        y_pred_reshaped = tf.reshape(y_pred, [-1, 20, 3]) * 100
        y_true_reshaped = tf.reshape(y_true, [-1, 20, 3]) * 100

        # Compute the Euclidean distance between predicted and true joint positions
        errors = tf.sqrt(tf.reduce_sum(tf.square(y_true_reshaped - y_pred_reshaped), axis=-1))

        # Compute MPJE for each sample in the batch
        mean_per_sample = tf.reduce_mean(errors, axis=-1)

        # Update total error and number of samples
        self.mpje.assign_add(tf.reduce_sum(mean_per_sample))
        self.num_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        # Compute mean per joint error across all samples
        return self.mpje / self.num_samples

    def reset_state(self):
        # Reset accumulated metrics
        self.mpje.assign(0)
        self.num_samples.assign(0)


def custom_metric(**kwargs):
    return MPJE()


def mpjpe_loss(y_true, y_pred):
    y_pred_reshaped = tf.reshape(y_pred, [-1, 20, 3])
    y_true_reshaped = tf.reshape(y_true, [-1, 20, 3])
    errors = tf.sqrt(tf.reduce_sum(tf.square(y_true_reshaped - y_pred_reshaped), axis=-1))
    return tf.reduce_mean(errors)


model = load_model('LSTM_model_noMove_epochs200_25102024',
                   custom_objects={'MPJE': custom_metric, 'mpjpe_loss': mpjpe_loss})


# rf_model = pickle.load(open('RF_model_new2.pkl', 'rb'))


# ------------------------------- Filtering --------------------------- #

def filtering(P, cutoff=1 / 60):
    a, b = signal.butter(3, cutoff, btype='low', analog=False)
    P_filtered = np.array(P)

    for coord in range(3):
        P_filtered[:, coord] = signal.filtfilt(a, b, P_filtered[:, coord])

    return P_filtered


# ------------------------------ MQTT ------------------------------ #

def connect_to_broker(broker_ip):
    client = mqtt.Client("FusionSync")
    client.on_connect = on_connect
    client.on_message = on_message
    if client.connect(broker_ip) != 0:
        print("Could not connect to MQTT broker")
    else:
        return client


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe("PoseEst/TIP")
        client.subscribe("PoseEst/MP")
    else:
        print(f"Failed to connect with code {rc}")


def on_message(client, userdata, message):
    try:
        payload = message.payload.decode()
        data = json.loads(payload)

        if message.topic == "PoseEst/TIP":
            sensor(data)
        elif message.topic == "PoseEst/MP":
            visuals(data)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")


# ------------------------------ Data Preproc ------------------------------ #
def visuals(visual_data):
    visual_ts = visual_data['timestamp']

    visualarray = np.asarray(visual_data['mediapipe_joints'])
    visual_joints = visualarray[indices, 0:3].reshape(len(indices) * 3, )

    for k in range(1, len(visual_joints), 3):
        visual_joints[k] = visual_joints[k] * (-1)

    visual_buffer.append((visual_joints, visual_ts))
    if len(visual_buffer) > max_buffer_size:
        visual_buffer.pop(0)

    # dataSync(visual_joints, visual_ts)
    # print('VISUAL TIME', visual_ts)
    dataSync()

    return


def sensor(sensor_data):
    global hips_data
    sensor_ts = sensor_data['timestamp']

    sensorarray = np.asarray(sensor_data['joint_data'])
    sensorarray2 = sensorarray[:, (1, 2, 0)] * (1, 1, -1)
    sensor_joints = sensorarray2.reshape(20, 3)
    hips = sensor_joints[0]
    hips_data.append(hips)
    sensor_joints = (sensor_joints - hips).reshape(60, )

    sensor_buffer.append((sensor_joints, sensor_ts))
    if len(sensor_buffer) > max_buffer_size:
        sensor_buffer.pop(0)

    # tip_data.append(sensor_joints)
    # tip_ts.append(sensor_ts)

    # print('SENSOR TIME', sensor_ts)

    return


# print('t0', t0.time(), 'ts', ts.time(), 't1', t1.time(), 'now', datetime.now(timezone.utc).strftime('%T.%f')[:-3])

# ------------------------------ Sync ------------------------------ #
def dataSync():
    # global t0, vis_moved

    if not sensor_buffer or not visual_buffer:
        return

    latest_sensor_joints, latest_sensor_ts = sensor_buffer[-1]

    # print('test')

    reversed_buffer = visual_buffer
    reversed_buffer.reverse()

    for i in range(len(reversed_buffer) - 1):
        visual_joints, visual_ts = reversed_buffer[i]
        _, previous_ts = reversed_buffer[i + 1]

        threshold = (dt.strptime(visual_ts, "%H:%M:%S.%f") - dt.strptime(previous_ts,
                                                                         "%H:%M:%S.%f")).total_seconds() * 1000

        if withinthreshold(latest_sensor_ts, visual_ts, threshold):
            print('true')
            input1 = np.concatenate((latest_sensor_joints, visual_joints))
            input_data.append(input1)
            break

    if len(input_data) == 5:
        fusion(input_data)
        input_data.pop(0)

    return


def withinthreshold(sensor_ts, visual_ts, threshold_ms=100.0):
    sensor_time = dt.strptime(sensor_ts, "%H:%M:%S.%f")
    visual_time = dt.strptime(visual_ts, "%H:%M:%S.%f")

    print((sensor_time - visual_time).total_seconds())

    # Check if within the threshold
    return abs((sensor_time - visual_time).total_seconds() * 1000) <= threshold_ms


def fusion(input):
    global hips_data
    array_for_fusion = np.array(input).reshape((1, 5, 99))

    # output = rf_model.predict(model.predict(array_for_fusion, verbose=0))
    output = model.predict(array_for_fusion, verbose=0)

    if len(hips_data) >= 24:
        hips_position = filtering(hips_data)[-1, :]
        hips_data.pop(0)
    else:
        hips_position = hips_data[-1]

    output = (output.reshape(20, 3) + hips_position).reshape(60, )

    print('FUSION', output)

    pred.append(output)
    animation(pred)

    return


# ------------------------------ Save txt Animation ------------------------------ #

def animation(predictions):
    with open("fusion_predictions.txt", "w") as file:
        for pred in predictions:
            pred_line = ' '.join(map(str, pred.flatten()))
            file.write(pred_line + '\n')


#  ------------------------------------------------------------------  #

def pickletodict(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


def main():
    print('Fusion Sync')
    broker_ip = 'localhost'
    client = connect_to_broker(broker_ip)
    if client:
        client.loop_forever()


main()
