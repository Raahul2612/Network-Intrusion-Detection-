# Network-Intrusion-Detection
Network intrusion detection using Random forest (ML model) and LSTM (DL model)

PROBLEM : 
 
In today’s rapidly evolving digital landscape, computer networks are increasingly vulnerable 
to a wide array of cyber-attacks and intrusions. Traditional security mechanisms like firewalls 
and antivirus programs are no longer sufficient to detect sophisticated and stealthy threats. 
This necessitates the development of intelligent systems capable of identifying abnormal 
activities in real-time and with high accuracy. 
The primary objective of this project is to design and evaluate a Network Intrusion Detection 
System (NIDS) using two distinct machine learning approaches: 
1. Random Forest, a robust ensemble-based classifier known for its accuracy and 
interpretability on structured data. 
2. Long Short-Term Memory (LSTM), a deep learning model well-suited for sequence 
and temporal pattern analysis in network traffic.

OBJECTIVE : 
 
1.   To analyse the behaviour of network traffic and detect anomalies by categorizing 
them into Normal, Malicious, and Suspicious classes using the NSL-KDD dataset. 
2.  To implement and compare two machine learning models for multi-class 
classification: 
• Random Forest (a robust ensemble-based model). 
• LSTM (Long Short-Term Memory) network (a sequential deep learning model suited 
for temporal traffic patterns). 
3.  To classify each instance of network traffic accurately into one of the three target 
classes: 
• Normal – legitimate user behaviour, 
• Malicious – confirmed attack behaviours (e.g., DoS, U2R), 
• Suspicious – potentially harmful or abnormal activities not fully confirmed as attacks. 
4. To evaluate the classification performance of both models based on accuracy 
5. To compare and interpret the results to determine which model is more effective for 
real-time network intrusion detection and can be reliably used in practical 
cybersecurity applications.

DATASET : 
 
1.  Source Files : 
 
• lstm_traffic_dataset.csv 
• rf_traffic_dataset.csv 
Both datasets are CSV files randomly generated , which is widely used for intrusion 
detection tasks. 
 
2. Number of records and features : 
  
NUMBER OF FEATURES 
lstm_traffic_dataset.csv :
NUMBER OF RECORDS : 320,500 
NUMBER OF FEATURES : 42
rf_dataset.csv 
NUMBER OF RECORDS : 120,500 
NUMBER OF FEATURES : 42
 
3. LABEL DESCRIPTION : 
 
Normal – Legitimate traffic with no malicious behaviour. 
 
Malicious – Clearly identified harmful activity such as DDoS, brute force, port scans, 
etc. 
 
Suspicious – Traffic that shows abnormal patterns, but not confidently classified as 
malicious (may represent zero-day attacks, misconfigurations, or unusual usage 
patterns). 
 
 
4. FEATURE DESCRIPTION : 
 
Basic Connection Features 
duration - Length of the connection in seconds. 
protocol_type - Type of network protocol (e.g., TCP, UDP, ICMP). 
service - Network service on the destination (e.g., HTTP, FTP, SMTP, DNS). 
flag - Status of the connection (e.g., SF, S0, RSTO). 
src_bytes - Number of bytes transferred from source to destination. 
dst_bytes - Number of bytes transferred from destination to source. 
 
Content-Based Features 
land - Whether source and destination addresses/ports are the same (1 = yes, 0 = no). 
wrong_fragment - Number of wrong fragments in the packet. 
urgent - Number of urgent packets in the connection. 
hot - Number of "hot" indicators (e.g., accessing sensitive files). 
num_failed_logins - Number of failed login attempts. 
logged_in - Whether the session is logged in (1 = yes, 0 = no). 
num_compromised - Number of compromised conditions. 
root_shell - Whether a root shell was obtained (1 = yes, 0 = no). 
su_attempted - Whether su command was attempted (1 = yes, 0 = no). 
num_root - Number of root accesses. 
num_file_creations - Number of file creation operations. 
num_shells - Number of shell prompts. 
num_access_files - Number of file access operations. 
num_outbound_cmds - Number of outbound commands (almost always 0). 
is_host_login - Whether the login is from a host (1 = yes, 0 = no). 
is_guest_login - Whether the login is as a guest user (1 = yes, 0 = no). 
 
Time-Based Traffic Features 
count - Number of connections to the same host in the past 2 seconds. 
srv_count - Number of connections to the same service in the past 2 seconds. 
serror_rate - Percentage of connections with SYN errors. 
srv_serror_rate - Percentage of connections to the same service with SYN errors. 
rerror_rate - Percentage of connections with REJ errors. 
srv_rerror_rate - Percentage of connections to the same service with REJ errors. 
same_srv_rate - Percentage of connections to the same service. 
diff_srv_rate - Percentage of connections to different services. 
srv_diff_host_rate - Percentage of connections to different hosts. 
 
Host-Based Traffic Features 
dst_host_count - Number of connections to the same destination host. 
dst_host_srv_count - Number of connections to the same service on the destination host. 
dst_host_same_srv_rate - Percentage of connections to the same service on the destination 
host. 
dst_host_diff_srv_rate - Percentage of connections to different services on the destination 
host. 
dst_host_same_src_port_rate - Percentage of connections to the same source port. 
dst_host_srv_diff_host_rate - Percentage of different hosts accessing the same service. 
dst_host_serror_rate - Percentage of connections with SYN errors to the destination host. 
dst_host_srv_serror_rate - Percentage of connections to the same service with SYN errors. 
dst_host_rerror_rate - Percentage of connections with REJ errors to the destination host. 
dst_host_srv_rerror_rate - Percentage of connections to the same service with REJ errors. 
Target feature - label - Classification of the connection as normal or malicious or suspicious . 
 
5. PRE-PROCESSING : 
 
Label Encoding is a preprocessing technique used to convert categorical (text) data 
into numeric values, so that machine learning algorithms can understand and process 
them efficiently. 
 
Categorial features like protocol_type , services and flag are converted to numeric 
values and also target features ( normal , malicious and suspicious ) are also converted 
to numeric values for efficiency . 
 
Example : Normal – 0 , Malicious – 1 and Suspicious – 2 . 

SOLUTION : 
 
What are you proposing as a novel solution to solve the problem? 
 
We propose a hybrid deep learning and machine learning-based intrusion detection system 
(IDS) that uses Long Short-Term Memory (LSTM) networks and Random Forest classifiers 
to accurately detect network intrusions in varying traffic scenarios. Our novel approach lies 
in: 
• Simulating real-world dynamic traffic by training models on different packet sizes 
(e.g., 50, 100, 150…). 
• Comparing temporal (LSTM) and non-temporal (Random Forest) learning models to 
identify the optimal method for intrusion detection in a streaming packet environment. 
• Providing insights into how data volume (packet size) influences IDS performance—
 this can help tune real-time systems for efficiency. 
 
How are you going to solve it? 
 
1. Preprocessing the  dataset : 
o Handle missing values, encode categorical features, and normalize numerical 
values. 
o Simulate streaming by varying packet sizes during model training. 
 
 
2. Model Training: 
o LSTM Model: Utilizes the sequential nature of packets to detect patterns over 
time. Ideal for uncovering deep temporal behaviours in the data. 
o Random Forest Model: Captures non-linear relationships in packet features 
quickly and efficiently, acting as a strong baseline. 
 
3. Evaluation & Comparison: 
o For each packet size , both models are trained and tested. 
o Metrics like accuracy and precision are computed and plotted to visualize 
performance trends. 
o This allows us to assess the scalability and adaptability of both models under 
realistic network traffic conditions. 
How are you using the dataset and AI to solve this? 
 
• Dataset: We use the dataset generated randomly with 42 features mentioned above , 
which contains labelled records of normal and malicious network connections. 
• AI Techniques: 
o LSTM (Deep Learning) is applied to learn sequence-based intrusion patterns. 
o Random Forest (Machine Learning) is used for fast and accurate classification 
without sequence learning. 
• Both models are trained on varying packet sizes to analyse how the amount of data 
affects detection accuracy, simulating different network load scenarios. 
 
Final Solution : 
 
The final solution is a flexible intrusion detection framework that: 
• Takes real-time network traffic input, 
• Preprocesses and feeds it into either an LSTM-based model (for deeper, sequence
aware detection) or a Random Forest model (for fast detection), 
• Adapts dynamically to network traffic volume (packet size), 
• Provides performance metrics (accuracy, precision) to help tune deployment settings. 
This approach enables organizations to choose the most efficient IDS model based on current 
network load, making cybersecurity systems more intelligent, adaptive, and accurate. 

ALGORITHM : 

1) RANDOM FOREST : 
 
Step 1: Load the dataset (CSV file with labelled network traffic) 
Step 2: Encode categorical features (protocol_type, service, flag) using 
LabelEncoder 
Step 3: Extract labels and encode them (normal, attack types) 
Step 4: Drop label column and scale features using StandardScaler 
Step 5: Split dataset into training and testing sets (80/20 split) 
Step 6: Train the Random Forest model (n_estimators=100) 
Step 7: Predict on test set 
Step 8: Evaluate model using accuracy score 
Step 9: Repeat training for different packet sizes to analyse performance 
Step 10: Plot accuracy for various packet sizes 
 
2) LSTM : 
  
 Step 1: Load dataset from CSV file 
    Step 2: Encode categorical columns: protocol_type, service, flag 
  Step 3: Encode target labels (label) and one-hot encode them 
  Step 4: Drop label column and apply StandardScaler to numerical     
features         
  Step 5: Reshape input features to LSTM shape: (samples, timesteps=1, 
features) 
  Step 6: Split data into training and test sets (80/20) 
  Step 7: Define LSTM model with two LSTM layers + Dropout + Dense 
(softmax output) 
  Step 8: Compile using Adam optimizer and categorical crossentropy 
  Step 9: Train model for 10 epochs, store validation accuracy 
  Step 10: Evaluate on test set using accuracy score 
  Step 11: Repeat training for various packet sizes 
  Step 12: Plot validation accuracy vs. packet sizes
