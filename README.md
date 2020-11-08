# Statistical estimator of network traffic

The library is dedicated to estimating statistical properties of packets grouped
 within a network flows or even a device, given a .pcap file and target object identifier. 

The identifier must be specified at .pcap processing stage, where packet-related
 features (packet size, inter-arrival time and direction) are extracted. For 
 example, to extract device-level stats you can try the following:

```
export PYTHONPATH=.

python pcap_parsing/main.py \
--pcapfile=traffic_dumps/iot_amazon_echo.pcap \
--identifier='44:65:0d:56:cc:d3'
```
To process a separate flow, do something like:
```
python pcap_parsing/main.py \
--pcapfile=traffic_dumps/skypeLANhome.pcap \
--identifier="UDP 192.168.0.102:18826 192.168.0.105:26454" \
--flow_level
```

Given the target stats, we need to train a gaussian mixture that 
maps a packet's features to a centroid value, effectively 
transforming initial features to discrete sequences.

```
python features/train_quantizer.py \
--dataset="traffic_dumps/iot_amazon_echo_44:65:0d:56:cc:d3.csv"
```

This allows us to easily use various sequence models, like Markov chains:
```
python markov_baseline/train_evaluate_markov.py \
--dataset="traffic_dumps/iot_amazon_echo_44:65:0d:56:cc:d3.csv" \
--quantizer_path="obj/iot_amazon_echo_44:65:0d:56:cc:d3"
```
or autoregressive neural networks, either recurrent (RNN) or temporal
convolutional networks (TCN):
```
python nn_generators/train_generator.py \
--dataset="traffic_dumps/iot_amazon_echo_44:65:0d:56:cc:d3.csv" \
--quantizer_path="obj/iot_amazon_echo_44:65:0d:56:cc:d3" \
--generator_name=RNN
```
 
## ITL paper

The code for the paper below is available at this 
[tag](https://github.com/RadionBik/Statistical-estimator-of-network-traffic/releases/tag/v0.1):
 
* Bikmukhamedov R., Nadeev A., Maione G., and Striccoli D., "Comparison of HMM and RNN
models for network traffic modeling", _Internet Technology Letters_, 2020. DOI: 10.1002/itl2.147
