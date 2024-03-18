# LAN

Code for paper "LAN: Learning Adaptive Neighbors for Real-Time ITD"

## Dataset Download

You can download dataset CERT r4.2 and r5.2 from  https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247


The expected project structure is:

```
LAN
 |-- run.sh
 |-- run.py
 |-- inference.py
 |-- model.py
 |-- utils.py
 |-- data
 |    |-- output
 |    |-- r4.2
 |    |    |-- ...  
 |    |-- r5.2
 |    |    |-- ...      
 |    |-- answers
 |    |    |-- ...  
```

## How to run
(Recommend)You can run `run.py` to train LAN.
You can run `inference.py` for inference

