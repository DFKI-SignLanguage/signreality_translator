import argparse
from xmlrpc.server import SimpleXMLRPCRequestHandler
from xmlrpc.server import SimpleXMLRPCServer

import os
import socket
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class StringDataset(Dataset):
    def __init__(self, strings):
        self.strings = strings

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        return self.strings[idx]


class Translator:
    def __init__(self, pretrained_model_filename, finetuned_model_filename):
        """
        Initialize the translation engine, by loading a saved model in memory, so that it waits for
        translation requests
        :param model_name: the name of the saved model
        :type model_name: str
        :param pretrained_model_filename: the checkpoint filename of the pre-trained model to load
        :type pretrained_model_filename: str
        :param save_file_path: the path where models were saved
        :type save_file_path: str
        :param fold: the numerical ide of the train fold to be loaded (default=1)
        :type fold: int
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading pretrained model...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_filename).to(self.device)

        print(f"Loading fine-tuned model...")
        # make sure the path is absolute
        if not os.path.isabs(finetuned_model_filename):
            abspath = os.path.dirname(os.path.abspath(__file__))
            finetuned_model_filename = os.path.join(abspath, finetuned_model_filename)

        print(f"Loading fine-tuned model...")
        self.model.load_state_dict(torch.load(finetuned_model_filename))
        self.model.eval()
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_filename)

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


def start_server(rpc_path, host, port, exposed_function, pretrained_model, finetuned_model):
    hostname = socket.gethostname()
    host = socket.gethostbyname(hostname)
    # Restrict to a particular path.
    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = (rpc_path, )

    # Create server
    with SimpleXMLRPCServer((host, port), requestHandler=RequestHandler) as server:
        server.register_introspection_functions()

    # Initialize the Translator class
    translator = Translator(pretrained_model, finetuned_model)

    # Register the translate function
    server.register_function(translator.translate, exposed_function)

    # Run the server's main loop
    print(f"Serving XML-RPC on {host} port {port}")
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start text-to-gloss translator XML-RPC server with config file.')
    path = os.path.abspath(__file__)
    parser.add_argument('--config', type=str, help='Path to the YAML config file',
                        default=os.path.join(path, 'config/translator.yaml'))
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)
    start_server(**params)
