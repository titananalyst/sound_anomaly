# Sound anomaly detection
A diagnostic model was developed and for this the extensive MIMII dataset, which provides a variety of audio data from industrial machines under different operating conditions, was used. In particular, we focus on the recordings of a pump. The aim is to develop a model that can detect faulty conditions of this pump.

A semi-supervised learning approach is used. The dataset contains labels for normal (healthy) and abnormal (faulty) data. An autoencoder was trained with only the normal data. The challenge is to detect anomalies in the test set, which contains both normal and abnormal data. The evaluation of anomalies depends on what data is considered normal. Based on this, a set of models was created based on different SNR levels of the pump data. In a further step, the data collected with the created prototype will be used.

