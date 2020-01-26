# Statistical estimator of network traffic

The library is dedicated to estimating statistical properties of netwrok flows or devices
 given a .pcap file. 

To begin, setup a virtual environment (Python 3.7), and start with exploring 
jupyter notebooks.

## Jupyter notebooks

The notebooks were converted via [Jupytext](https://github.com/mwouts/jupytext) to 
`.py` format and stored within the 
`jupytext_notebooks` folder. To restore jupyter notebooks (`.ipynb`), execute:

    jupytext --to notebook jupytext_notebooks/ITL_paper.py 


The most important notebook is `ITL_paper.py` that contains the code implementing results
from the following paper:
 
* Bikmukhamedov R., Nadeev A., Maione G., and Striccoli D., "Comparison of HMM and RNN
models for network traffic modeling", _Internet Technology Letters_, 2020. DOI: 10.1002/itl2.147

 
## Imitator

This repo was developed around idea to develop software that would allow to imitate
any given device or a flow. At this point, the algorithms used in the estimator itself
are a bit obsolote, so for the freshest ones check out the notebooks provided above. 

The estimator consists of 2 main parts: estimator and generator.

### Traffic estimator

The estimator takes a pcap file and for all (or those specified in `addresses_to_check.txt`) 
flows extracts the Payload Length and the Inter-Arrival Time parameters. 
Afterwards, the EM-algorithm is applied to fit a gaussian mixture (up to 5 components) 
to each parameter and the trained model is saved to a file in the `obj/` folder.

Usage examples:
    
    python packet_estimator.py -p traffic_dumps/skypeLANhome.pcap -i flow --plot

    python packet_estimator.py -p traffic_dumps/skypeLANhome.pcap -i flow -f all -n auto


### Traffic generator

The generator uses the saved EM-model to generate packets with length and delay drawn from the estimated distributions. The side has to be specified, either 'local' (the estimated side) or 'remote' (remote end).   

Example run for the local side:

    python packet_transceiver.py  -m local -i wlp2s0 -d 192.168.88.20 -o obj/skypeLANhome__auto_components.pkl

.. and the remote:

    python packet_transceiver.py -m remote -i wlp2s0 -d 192.168.88.10 -o obj/skypeLANhome__auto_components.pkl
   
