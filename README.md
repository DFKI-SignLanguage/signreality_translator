# Text to gloss machine translation

This is a server that allows translations from text to sign language glosses. The server loads a pre-trained translation model compatible to the sequence-to-sequence Huggingface interface and exposes a translate function as an XMLRPC server. 

## Requirements

A machine with a GPU and 32 GB of RAM. 


## Installation

1. Clone this code on your computer
```
git clone https://github.com/DFKI-SignLanguage/text-to-gloss-machine-translation.git .
```
2. Go to the sub-directory named "models" and download the translation model
```
cd models
wget https://cloud-affective.dfki.de/s/SJCWfKdwfYDTTQL/download/result_fold_0_best_model.pt 
cd ..
```
3. Create a virtual machine and install the requirements
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
4. Start the translation server
```
venv/bin/python translator.py
```



## Usage

The XML-RPC server can be accessed by any compatible client. An example on how to use from a commandline can be found here:

```
curl -s \
-H "Content-Type: text/xml" \
-d '<?xml version="1.0"?>
<methodCall>
<methodName>translate</methodName>
<params>
<param>
<value><string>der zug ICE 234 aus mainz fährt um 4 : 40 Uhr .</string></value>
</param>
</params>
</methodCall>' \
http://localhost:8000/RPC2
```

Please not that the text needs to be lowercased. 

Alternatively, you can try directly from within Python code with the provided example:

    python translator_client.py


## Configuration

The translator script takes the parameter `--config` and allows parameters to be specified in a YAML configuration file. Two sample configuration files are provided. Example:

```
venv/bin/python translator.py --config config/translator.yaml
```

The following parameters can be configured:
 - __pretrained model__: the name of the huggingface pre-trained model that needs to be loaded. By default it is `facebook/nllb-200-distilled-600M`. It doesn't need to be downloaded.
 - __finetuned_model__: the fine-tuned model that needs to be downloaded as instructed above. By default it resides in the subdirectory `model`
 - __port__: the port where the server shall run
 - __rpc_path__: the rpc path that the server will respond to
 - __exposed_function__: the name of the function that will be exposed to XML_RPC

There is also the commandline option `--listen-all-interfaces`.  If true, theserver will listen on all network interfaces, otherwise only on localhost.


## Credits

Based on the models by Maithri Rao, expanded by Eleftherios Avramidis, with the help of Cristina Espana Bonet and Fabrizio Nunnari. German Research Center for Artificial Intelligence (DFKI GbmbH).

Initial work funded by the research project BIGECO (BMBF, 2023-2026). Adaptation to fit the extended reality animation engine done by SignReality (EU 101070631 - UTTER open call: Development and application of deep models for eXtended Reality).  

