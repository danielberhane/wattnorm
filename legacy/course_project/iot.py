import boto3
import pandas as pd
import json
import time
from datetime import datetime
from awscrt import mqtt
from awsiot import mqtt_connection_builder

# Load CSV files
demand_df = pd.read_csv('your_demand.csv')

# AWS IoT Core settings
endpoint = "your-iot-endpoint"  # Get from IoT Core > Settings
cert_path = "path/to/certificate.pem.crt"
pri_key_path = "path/to/private.pem.key"
ca_path = "path/to/AmazonRootCA1.pem"
client_id = "building-sensor"
topic = "building/sensors"

# Create MQTT connection
mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=endpoint,
    cert_filepath=cert_path,
    pri_key_filepath=pri_key_path,
    ca_filepath=ca_path,
    client_id=client_id
)

print("Connecting to AWS IoT Core...")
connect_future = mqtt_connection.connect()
connect_future.result()
print("Connected!")

# Publish data
try:
    for index, row in demand_df.iterrows():
        message = {
            "timestamp": str(row['Timestamp']),
            "demand": float(row['Demand'])
        }
        
        mqtt_connection.publish(
            topic=topic,
            payload=json.dumps(message),
            qos=mqtt.QoS.AT_LEAST_ONCE
        )
        print(f"Published: {message}")
        time.sleep(1)  # Delay between messages

finally:
    disconnect_future = mqtt_connection.disconnect()
    disconnect_future.result()
