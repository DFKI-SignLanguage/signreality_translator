
import math
import torch
import torch.nn as nn
from collections import Counter
from torch import Tensor
import io
import time
import os
import pandas as pd
import json
from datetime import datetime

features_names = ["maingloss", "domgloss", "ndomgloss", "domreloc", "ndomreloc",
                  "domhandrelocx", "domhandrelocy", "domhandrelocz", "domhandrelocax", 
                  "domhandrelocay", "domhandrelocaz", "domhandrelocsx", "domhandrelocsy", "domhandrelocsz",
                  "domhandrotx", "domhandroty", "domhandrotz", 
                  "ndomhandrelocx", "ndomhandrelocy", "ndomhandrelocz", "ndomhandrelocax", 
                  "ndomhandrelocay", "ndomhandrelocaz", "ndomhandrelocsx", "ndomhandrelocsy", "ndomhandrelocsz",
                  "ndomhandrotx", "ndomhandroty", "ndomhandrotz"]

def read(text_info, mms_info):
    data_list = []
    (text_directory, text_encoding) = text_info
    print("text_directory: ", text_directory)
    (mms_directory, mms_encoding) = mms_info
    for filenumber in os.listdir(text_directory):
        f = os.path.join(mms_directory, filenumber+".mms")
        try:
            df = pd.read_csv(f, encoding=mms_encoding)
        except FileNotFoundError as e:
            print(f"WARNING: Text file exists while mms file does not, skipping: {e}")
            continue   

        text_address = os.path.join(text_directory, filenumber, "gebaerdler.Text_Deutsch.annotation~")
        file = open(text_address, encoding=text_encoding)
        lines = file.readlines()
        text_line = ""
        for i, text_data in enumerate(lines):
            if i>0:
                text_line = text_line + " " + text_data.replace("\n", "").split(";")[2] 
            else:
                text_line = text_line + text_data.replace("\n", "").split(";")[2]        
        
        data_dict = {"file_ID":filenumber, "text": text_line}
        for feature in features_names:
            if feature == "domgloss" or feature == "ndomgloss":
                temp = df[feature].copy()
                data_dict[feature] = [data_dict["maingloss"][i] if pd.isnull(token) else token for i,token in enumerate(temp)]
            else:
                data_dict[feature] = df[feature].tolist()
        data_list.append(data_dict)
    return data_list    