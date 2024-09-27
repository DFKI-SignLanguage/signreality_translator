import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import os
from sacrebleu.metrics import BLEU
from . import datasets
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import time
from enum import Enum, verify, UNIQUE

CHECKPOINT = 'facebook/nllb-200-distilled-600M' #for nllb

class StringDataset(Dataset):
    def __init__(self, strings):
        self.strings = strings

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        return self.strings[idx]


class Translator:
    def __init__(self, model_name, checkpoint, save_file_path, fold=1):
        """
        Initialize the translation engine, by loading a saved model in memory, so that it waits for
        translation requests
        :param model_name: the name of the saved model
        :type model_name: str
        :param checkpoint: the checkpoint filename of the model to load
        :type checkpoint: str
        :param save_file_path: the path where models were saved
        :type save_file_path: str
        :param fold: the numerical ide of the train fold to be loaded (default=1)
        :type fold: int
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)
        self.model.load_state_dict(torch.load(save_file_path + f"_fold_{fold}_{model_name}"))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def translate(self, strings):
        """

        :param strings:
        :return:
        """
        translation_output = []

        # Batch the input strings
        dataset = StringDataset(strings)
        test_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        with torch.no_grad():
            for text_tokens_padded, in test_dataloader:
                text_tokens_padded = text_tokens_padded.to(self.device)
                max_length = int(round(text_tokens_padded.size(1) * 1.5, 0))
                model_response = self.model.generate(input_ids=text_tokens_padded,
                                                     max_length=max_length)
                for i in range(text_tokens_padded.size(0)):
                    text_predicted = self.tokenizer.decode(model_response[i], skip_special_tokens=True)
                    translation_output.append(text_predicted)
        return translation_output


def start_server(**params):
    from xmlrpc.server import SimpleXMLRPCServer
    from xmlrpc.server import SimpleXMLRPCRequestHandler

    # Restrict to a particular path.
    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ('/RPC2',)

    # Create server
    with SimpleXMLRPCServer(('localhost', 8000), requestHandler=RequestHandler) as server:
        server.register_introspection_functions()

    # Initialize the Translator class
    translator = Translator(**params)

    # Register the translate function
    server.register_function(translator.translate, 'translate')

    # Run the server's main loop
    print("Serving XML-RPC on localhost port 8000")
    server.serve_forever()

if __name__ == '__main__':
    start_server()