import argparse
from xmlrpc.server import SimpleXMLRPCRequestHandler
from xmlrpc.server import SimpleXMLRPCServer

import os
import socket
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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

        self.model.load_state_dict(torch.load(finetuned_model_filename))
        self.model.eval()
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_filename)

    def translate(self, sentence):
        """

        :param strings:
        :return:
        """

        # Tokenize the input sentence
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Generate translation
        translated_tokens = self.model.generate(**inputs)

        # Decode the translated tokens
        translated_sentence = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_sentence


def start_server(listen_all_ifaces: bool, rpc_path, port, exposed_function, pretrained_model, finetuned_model):

    if listen_all_ifaces:
        host = ""
    else:
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


#
#
if __name__ == '__main__':

    path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Start text-to-gloss translator XML-RPC server with config file.')
    parser.add_argument('--config', type=str, help='Path to the YAML config file',
                        default=os.path.join(path, 'config/translator.yaml'))
    parser.add_argument('--listen-all-interfaces', action="store_true",
                        help='If true, the server will listen on all network interfaces, otherwise only on localhost.')

    args = parser.parse_args()

    listen_all_ifaces = args.listen_all_interfaces

    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    start_server(listen_all_ifaces, **params)
