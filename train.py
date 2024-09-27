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
from torch.utils.data import DataLoader
import time
from enum import Enum, verify, UNIQUE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = 'facebook/nllb-200-distilled-600M' #for nllb
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

@verify(UNIQUE)
class Translation(Enum):
    TEXT_TO_GLOSS = 0
    GLOSS_TO_TEXT = 1

def train(fold, ds, translation, augment):

    if translation == Translation.TEXT_TO_GLOSS:
        translation_dir = "textTogloss"
    elif translation == Translation.GLOSS_TO_TEXT:
        translation_dir = "glossTotext"
    else:
        raise ValueError("Invalid translation ")

    if not augment:
        augment_dir = "original_data"
    else:
        augment_dir = "aug_data"
    save_folder = os.path.join("/ds/videos/AVASAG/k_fold1/", translation_dir, augment_dir, "nllb")
    save_file_path = os.path.join(save_folder, "result")
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    (original, modified, full) = ds
    dataset = original

    # Split the dataset into 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(kf.split(dataset))

    # Split the dataset into train and test sets based on the current fold
    train_indices = [idx for fold_idx, idx in enumerate(folds[fold][0]) if fold_idx != fold]
    test_indices = folds[fold][1]
    train_data = [dataset[idx] for idx in train_indices]
    test_data = [dataset[idx] for idx in test_indices]

    # Augment the training data if augment=True
    if augment:
        train_data = augment_data(train_data, modified)
        
    train_dataset = datasets.SignLanguageDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=datasets.collate_fn)

    test_dataset = datasets.SignLanguageDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=datasets.collate_fn)

    NUM_EPOCHS = 1000
    loss_graf = []

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_log = open(save_file_path+ f"_fold_{fold}_train_log.txt", 'w')

    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, translation, tokenizer)

        end_time = time.time()
        log = "Epoch: " + str(epoch)+", Train loss: "+ str(train_loss)+" Epoch duration "+ str(end_time - start_time)+"\n"
        train_log.write(log)
        if epoch <= 1 or train_loss < min(loss_graf):
            best_model_path = save_file_path+f"_fold_{fold}_best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            log = "min so far is at epoch: "+ str(epoch)+"\n"
            train_log.write(log)
            best_epoch = epoch

        loss_graf.append(train_loss)

    log = "best epoch is: "+ str(best_epoch)
    train_log.write(log)
    train_log.close()

    torch.save(model.state_dict(), save_file_path+f"_fold_{fold}_last_model.pt")

    return test_dataloader, save_file_path


def evaluate(fold, model_name, test_dataloader, save_file_path, translation): # Evaluation
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.load_state_dict(torch.load(save_file_path+f"_fold_{fold}_{model_name}"))

    ground_truth = []
    hypothesis = []
    num_P_T = 0
    num_T_P = 0
    num_e = 0

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            file_Id, text_tokens_padded, maingloss_tokens_padded = batch
            text_tokens_padded = text_tokens_padded.to(device)
            maingloss_tokens_padded = maingloss_tokens_padded.to(device)

            if translation == Translation.TEXT_TO_GLOSS:
                input = text_tokens_padded
                output = maingloss_tokens_padded
            elif translation == Translation.GLOSS_TO_TEXT:
                input = maingloss_tokens_padded
                output = text_tokens_padded
            else:
                raise ValueError("Invalid translation ")

            #gloss to text
            outputs  = model(input_ids=input, labels=output)
            pred = model.generate(input_ids=input, max_length=output.size(1))

            for i in range(text_tokens_padded.size(0)):
                gt_maingloss = "".join(tokenizer.decode(maingloss_tokens_padded[i], skip_special_tokens=True))
                input_text = tokenizer.decode(text_tokens_padded[i], skip_special_tokens=True)
                text_predicted = tokenizer.decode(pred[i], skip_special_tokens=True)

                if fold == 9:

                    print(f"\nSample {len(ground_truth) + 1}:")
                    print(f"Prediction : {text_predicted}")

                    if translation == Translation.TEXT_TO_GLOSS:
                        print(f"Input Text: {input_text}")
                        print(f"Ground Truth Gloss: {gt_maingloss}")                    
                        ground_truth.append(gt_maingloss)
                
                    elif translation == Translation.GLOSS_TO_TEXT:
                        print(f"Input gloss: {gt_maingloss}")
                        print(f"Ground Truth text: {input_text}")
                        ground_truth.append(input_text)
                    else:
                        raise ValueError("Invalid translation ")

                    hypothesis.append(text_predicted)

                else:

                    # print(f"\nSample {len(ground_truth) + 1}:")
                    # print(f"Prediction : {text_predicted}")

                    if translation == Translation.TEXT_TO_GLOSS:
                        # print(f"Input Text: {input_text}")
                        # print(f"Ground Truth Gloss: {gt_maingloss}")                    
                        ground_truth.append(gt_maingloss)
                
                    elif translation == Translation.GLOSS_TO_TEXT:
                        # print(f"Input gloss: {gt_maingloss}")
                        # print(f"Ground Truth text: {input_text}")
                        ground_truth.append(input_text)
                    else:
                        raise ValueError("Invalid translation ")

                    hypothesis.append(text_predicted)

    # Calculate BLEU score    
    bleu = BLEU()
    result = bleu.corpus_score(hypothesis, [ground_truth])

    # Count sequence length comparisons
    num_P_T = sum(len(h.split()) > len(g.split()) for h, g in zip(hypothesis, ground_truth))
    num_T_P = sum(len(h.split()) < len(g.split()) for h, g in zip(hypothesis, ground_truth))
    num_e = sum(len(h.split()) == len(g.split()) for h, g in zip(hypothesis, ground_truth))

    # print(f"Predicted length > True length: {num_P_T}")
    # print(f"True length > Predicted length: {num_T_P}")
    # print(f"Equal lengths: {num_e}")

    # Save results to file
    with open(save_file_path + f"_fold_{fold}_outputs.txt", "w") as f:
        f.write(f"P>T: {num_P_T}\n")
        f.write(f"T>P: {num_T_P}\n")
        f.write(f"equal: {num_e}\n")
        f.write(f"BLEU score : {result.score}\n\n")

        for i, (gt, pred) in enumerate(zip(ground_truth, hypothesis)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Ground Truth Text: {gt}\n")
            f.write(f"Predicted Text: {pred}\n\n") 

    return result.score

def augment_data(train_data, sentences):

    augmented_train_data = train_data.copy()
    augmented_train_data.extend(sentences)

    return augmented_train_data  

def train_epoch(model, train_dataloader, optimizer, translation, tokenizer):
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        file_Id, text_tokens_padded, maingloss_tokens_padded = batch
        text_tokens_padded = text_tokens_padded.to(device)
        maingloss_tokens_padded = maingloss_tokens_padded.to(device)

        if translation == Translation.TEXT_TO_GLOSS:
            input_attention_mask = (text_tokens_padded != tokenizer.pad_token_id).to(device)
            input = text_tokens_padded
            output = maingloss_tokens_padded
        elif translation == Translation.GLOSS_TO_TEXT:
            input_attention_mask = (maingloss_tokens_padded != tokenizer.pad_token_id).to(device)
            input = maingloss_tokens_padded
            output = text_tokens_padded
        else:
            raise ValueError("Invalid translation ")   

        optimizer.zero_grad()

        output_final  = model(input_ids=input, attention_mask=input_attention_mask, labels=output)
        loss = output_final.loss    
        loss.backward()
        optimizer.step()

    avg_train_loss = loss / len(train_dataloader)     
    return avg_train_loss

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python train.py [--textTogloss|--glossTotext]")
        sys.exit(1)

    if sys.argv[1] == "--textTogloss":
        print("Using textTogloss Translation")
        translation = Translation.TEXT_TO_GLOSS
    elif sys.argv[1] == "--glossTotext":
        print("Using glossTotext Translation")
        translation = Translation.GLOSS_TO_TEXT
    else:
        print("You have to specify either textTogloss or glossTotext as an argument.")
        sys.exit(1)

    original_scores_best = []
    original_scores_last = []
    augmented_scores_best = []
    augmented_scores_last = []

    ds = datasets.read()    

    for fold in range(10):
        print(f"Current fold {fold}:")
        print("Original data :")
        test_dataloader, save_file_path = train(fold, ds, translation, augment=False)
        print("best model:")
        origina_score_best_model = evaluate(fold, "best_model.pt", test_dataloader, save_file_path, translation)
        print("last model:")
        original_score_last_model = evaluate(fold, "last_model.pt", test_dataloader, save_file_path, translation)
        original_scores_best.append(origina_score_best_model)
        original_scores_last.append(original_score_last_model)

        print("Augmented data:")
        test_dataloader, save_file_path = train(fold, ds, translation, augment=True)
        print("best model:")
        aug_score_best_model = evaluate(fold, "best_model.pt", test_dataloader, save_file_path, translation)
        augmented_scores_best.append(aug_score_best_model)
        print("last model:")
        aug_score_last_model = evaluate(fold, "last_model.pt", test_dataloader, save_file_path, translation)
        augmented_scores_last.append(aug_score_last_model)

    avg_original_score_best = np.mean(original_scores_best)
    avg_original_score_last = np.mean(original_scores_last)

    avg_augmented_score_best = np.mean(augmented_scores_best)
    avg_augmented_score_last = np.mean(augmented_scores_last)

    if translation == Translation.TEXT_TO_GLOSS:
        translation_str = "Text-Gloss"
    elif translation == Translation.GLOSS_TO_TEXT:
        translation_str = "Gloss-Text"
    else:
        raise ValueError("Invalid translation value")

    print(f"{translation_str} BLEU score on original data for each fold best_model: {original_scores_best}")
    print(f"{translation_str} BLEU score on original data for each fold last_model: {original_scores_last}")

    print(f"{translation_str} Average BLEU score on original data best_model: {avg_original_score_best}")    
    print(f"{translation_str} Average BLEU score on original data last_model: {avg_original_score_last}")

    print(f"{translation_str} BLEU score on augmented data for each fold best_model: {augmented_scores_best}")
    print(f"{translation_str} BLEU score on augmented data for each fold last_model: {augmented_scores_last}")

    print(f"{translation_str} Average BLEU score on augmented data best_model: {avg_augmented_score_best}")
    print(f"{translation_str} Average BLEU score on augmented data last_model: {avg_augmented_score_last}")