
import os
import json
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.func import read_data_non_linear
from non_linear_notebooks.run_eval_type_clean import run_full_attention_eval 
import logging
import csv

def change_logging_config(logging_path):
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Reconfigure logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO (use DEBUG for more verbosity)
        format='%(message)s',
        handlers=[
            logging.StreamHandler(),  # Print to console
            logging.FileHandler(logging_path, mode='a')  # Write to a file (append mode)
        ]
    )

import json
import numpy as np
import os
import logging

def evaluate_any_dataset(model_name="LLaVA-7B", dataset_name="POPE", prompt="oe", tau=False, logits_used=10, trained_model_path=None, config_list=None):
    # lr = config_list[0].get('lr', 1e-4)
    logging_path = f'./attention_results/eval_{prompt}_{dataset_name}_logits_used={logits_used}.log'
    csv_file = f'./attention_results/eval_{prompt}_{dataset_name}_logits_used={logits_used}.csv'
    if os.path.exists(logging_path):
        os.remove(logging_path)
    change_logging_config(logging_path)

    val_data, x_val_raw, _, x_0_val_raw = read_data_non_linear(
        model_name, dataset_name, split="val", prompt=prompt, token_idx=-2, return_data=True, type_l='_se'
    )

    try:
        train_data, x_train_raw, _, x_0_train_raw = read_data_non_linear(
            model_name, dataset_name, split="train", prompt=prompt, token_idx=-2, return_data=True, type_l='_se'
        )
        has_train_data = True
    except FileNotFoundError:
        print(f"Training data file not found for {dataset_name}. Skipping training data load since we are in evaluation mode.")
        has_train_data = False

    def post_process_sequences(x_raw, x_0_raw, logits_used):
        x_list, x_0_list = [], []
        for matrix_se, matrix_base in zip(x_raw, x_0_raw):
            if matrix_se.shape[0] < logits_used:
                padding = np.zeros((logits_used - matrix_se.shape[0], matrix_se.shape[1]), dtype=matrix_se.dtype)
                matrix_se = np.concatenate([matrix_se, padding], axis=0)
            else:
                matrix_se = matrix_se[:logits_used, :]
            
            x_0_vector = matrix_base[0, :] if matrix_base.ndim > 1 else matrix_base
            x_list.append(matrix_se)
            x_0_list.append(x_0_vector)
        return np.array(x_list), np.array(x_0_list)

    print(f"Processing dense multi-token sequence timelines for {dataset_name} directly from memory...")
    x_val, x_0_val = post_process_sequences(x_val_raw, x_0_val_raw, logits_used)
    
    if has_train_data:
        x_train, x_0_train = post_process_sequences(x_train_raw, x_0_train_raw, logits_used)
    else:
        x_train = np.empty((0, logits_used, x_val.shape[2])) if len(x_val) > 0 else np.array([])
        x_0_train = np.empty((0, x_0_val.shape[1])) if len(x_0_val) > 0 else np.array([])


    if prompt == "oe":
        val_labeled_path = f'./output/{model_name}/{dataset_name}_val_oe_labeled.jsonl'
        if os.path.exists(val_labeled_path):
            val_labeled = json.load(open(val_labeled_path))
            pred_list_val = [0 if ins["is_answer"] == 'no' else 1 for ins in val_labeled]
        else:
            print(f"Parsing base '{prompt}' strings dynamically via inline string matching...")
            pred_list_val = [0 if (ins['response'].lower().startswith("no") or "incorrect" in ins['response'].lower() or "not correct" in ins['response'].lower()) else 1 for ins in val_data]
        
        if has_train_data:
            train_labeled_path = f'./output/{model_name}/{dataset_name}_train_oe_labeled.jsonl'
            train_labeled = json.load(open(train_labeled_path)) if os.path.exists(train_labeled_path) else train_data
            pred_list_train = [0 if ins["is_answer"] == 'no' else 1 for ins in train_labeled] if os.path.exists(train_labeled_path) else [0 if (ins['response'].lower().startswith("no") or "incorrect" in ins['response'].lower() or "not correct" in ins['response'].lower()) else 1 for ins in train_data]

    elif prompt == "oeh":
        pred_list_val = [0 if ins['response'].startswith("Sorry, I cannot answer your question") else 1 for ins in val_data]
        if has_train_data: pred_list_train = [0 if ins['response'].startswith("Sorry, I cannot answer your question") else 1 for ins in train_data]
    elif prompt == "mq":
        pred_list_val = [0 if "no" in ins['response'].lower() else 1 for ins in val_data]
        if has_train_data: pred_list_train = [0 if "no" in ins['response'].lower() else 1 for ins in train_data]

    label_list_val = [ins['label'] for ins in val_data]
    correctness_list_val = [1 if label==pred else 0 for label, pred in zip(label_list_val, pred_list_val)]
    pred_list_val_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] else 0 for ins in val_data]
    label_list_val_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_val, pred_list_val_2)]
    y_val = np.array(label_list_val_2) 
    y_pred = np.array(pred_list_val_2)

    if has_train_data:
        label_list_train = [ins['label'] for ins in train_data]
        correctness_list_train = [1 if label==pred else 0 for label, pred in zip(label_list_train, pred_list_train)]
        pred_list_train_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] else 0 for ins in train_data]
        label_list_train_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_train, pred_list_train_2)]
        y_train = np.array(label_list_train_2) 
    else:
        y_train = np.array([])

    print(f"Label distribution for {dataset_name} Validation Split:", np.bincount(y_val))
    print("Type II label rate:", y_val.mean())

    run_full_attention_eval(x_train, 1 - y_train, x_val, 1 - y_val, model_name, dataset_name, prompt, csv_file, 1 - y_pred, tau=tau, x_0_train=x_0_train, 
                            x_0_val=x_0_val, type_num='2', logits_used=logits_used, trained_model_path=trained_model_path, config_list=config_list)


def evaluate_mad_2(model_name="LLaVA-7B", dataset_name="MAD", prompt="oe", tau=False, logits_used=10, trained_model_path=None, config_list=None):
    lr = config_list[0].get('lr', 1e-4)

    logging_path = f'./attention_results/{model_name}_{prompt}_{dataset_name}_type_II_l{logits_used}_lr{lr}.log'
    csv_file = f'./attention_results/{model_name}_{prompt}_{dataset_name}_type_II_l{logits_used}_lr{lr}.csv'
    if os.path.exists(logging_path):
        os.remove(logging_path)
    change_logging_config(logging_path)

    train_data, x_train_raw, _, x_0_train_raw = read_data_non_linear(
        model_name, dataset_name, split="train", prompt=prompt, token_idx=-2, return_data=True, type_l='_se'
    )
    val_data, x_val_raw, _, x_0_val_raw = read_data_non_linear(
        model_name, dataset_name, split="val", prompt=prompt, token_idx=-2, return_data=True, type_l='_se'
    )

    def post_process_sequences(x_raw, x_0_raw, logits_used):
        x_list = []
        x_0_list = []
        
        for matrix_se, matrix_base in zip(x_raw, x_0_raw):
            # matrix_se = np.nan_to_num(matrix_se, nan=0.0, posinf=100.0, neginf=-100.0)
            # matrix_base = np.nan_to_num(matrix_base, nan=0.0, posinf=100.0, neginf=-100.0)
            
            if matrix_se.shape[0] < logits_used:
                padding = np.zeros((logits_used - matrix_se.shape[0], matrix_se.shape[1]), dtype=matrix_se.dtype)
                matrix_se = np.concatenate([matrix_se, padding], axis=0)
            else:
                matrix_se = matrix_se[:logits_used, :]
            
            x_0_vector = matrix_base[0, :] if matrix_base.ndim > 1 else matrix_base
            
            x_list.append(matrix_se)
            x_0_list.append(x_0_vector)
            
        return np.array(x_list), np.array(x_0_list)

    print("Processing dense 3D multi-token sequence timelines directly from memory tensors...")
    x_train, x_0_train = post_process_sequences(x_train_raw, x_0_train_raw, logits_used)
    x_val, x_0_val = post_process_sequences(x_val_raw, x_0_val_raw, logits_used)


    if prompt == "oe":
        val_labeled_path = f'./output/{model_name}/MAD_val_oe_labeled.jsonl'
        if os.path.exists(val_labeled_path):
            val_labeled = json.load(open(f'./output/{model_name}/MAD_val_oe_labeled.jsonl')) #answers to first question
            train_labeled = json.load(open(f'./output/{model_name}/MAD_train_oe_labeled.jsonl'))
            pred_list_val = [0 if ins["is_answer"] == 'no' else 1 for ins in val_labeled]
            pred_list_train = [0 if ins["is_answer"] == 'no' else 1 for ins in train_labeled]
        else:
            print("Parsing base 'oe' strings dynamically via inline string matching...")
            # Rule-based validation: If it starts with "No" or says "incorrect"/"not correct", treat it as a 0 choice.
            pred_list_val = [0 if (ins['response'].lower().startswith("no") or "incorrect" in ins['response'].lower() or "not correct" in ins['response'].lower()) else 1 for ins in val_data]
            pred_list_train = [0 if (ins['response'].lower().startswith("no") or "incorrect" in ins['response'].lower() or "not correct" in ins['response'].lower()) else 1 for ins in train_data]
    elif prompt == "oeh":
        pred_list_val = [0 if ins['response'].startswith("Sorry, I cannot answer your question") else 1 for ins in val_data]
        pred_list_train = [0 if ins['response'].startswith("Sorry, I cannot answer your question") else 1 for ins in train_data]
    elif prompt == "mq":
        pred_list_val = [0 if "no" in ins['response'].lower() else 1 for ins in val_data]
        pred_list_train = [0 if "no" in ins['response'].lower() else 1 for ins in train_data]

    label_list_val = [ins['label'] for ins in val_data]
    label_list_train = [ins['label'] for ins in train_data]
    correctness_list_val = [1 if label==pred else 0 for label, pred in zip(label_list_val,pred_list_val)]
    correctness_list_train = [1 if label==pred else 0 for label, pred in zip(label_list_train,pred_list_train)]


    pred_list_val_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] else 0 for ins in val_data]
    pred_list_train_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] else 0 for ins in train_data]

    label_list_val_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_val,pred_list_val_2)]
    label_list_train_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_train,pred_list_train_2)]

    y_val = np.array(label_list_val_2) #Type II labels Was the model correct in determining if it was right or wrong?
    y_train = np.array(label_list_train_2) #Type II labels
    
    from sklearn.metrics import roc_auc_score
    print("Label distribution:", np.bincount(y_val))
    print("Type II label rate:", y_val.mean())

    y_pred = np.array(pred_list_val_2)
    run_full_attention_eval(x_train, 1 - y_train, x_val, 1 - y_val, model_name, dataset_name, prompt, csv_file, 1 - y_pred, tau=tau, x_0_train=x_0_train, 
                            x_0_val=x_0_val, type_num='2', logits_used=logits_used, trained_model_path=trained_model_path, config_list=config_list)
   

# OPTIONALY: Get the file path passed as an argument
# temp_file_path = sys.argv[1]
# def process_csv_data(file_path):
#     configs = []
#     with open(file_path, newline='') as f:
#         reader = csv.DictReader(f)   # automatically skips the header
#         for row in reader:
#             configs.append({
#                 'embed_dim':  int(row['embed_dim']),
#                 'num_heads':  int(row['num_heads']),
#                 'num_layers': int(row['num_layers']),
#                 'dropout':    float(row['dropout']),
#                 'lr':         float(row['lr']),
#                 'csv_file':   os.path.basename(file_path)
#             })
#     prompt = configs[0]['csv_file'].split('_')[-4]
#     csv_file = configs[0]['csv_file']
#     if "mPLUG-Owl" in csv_file:
#         model_name = 'mPLUG-Owl'
#     elif "LLaVA-7B" in csv_file:
#         model_name = "LLaVA-7B"
#     elif "MiniGPT4" in csv_file:
#         model_name = 'MiniGPT4'
#     elif "LLaMA_Adapter" in csv_file:
#         model_name = "LLaMA_Adapter"
#     else:
#         model_name = None
#     return configs, prompt, model_name

# config_list, prompt, model_name = process_csv_data(temp_file_path)

# config_list = [{'embed_dim':1024, 'num_heads':64, 'num_layers':2, 'dropout':0.5, 'epochs':100, 'lr':5e-7, 'log_reg':False}]
# config_list = [{'embed_dim':256, 'num_heads':8, 'num_layers':2, 'dropout':0.5, 'epochs':100, 'lr':5e-7, 'log_reg':False}]

# config_list = [{'embed_dim':1024, 'num_heads':64, 'num_layers':2, 'dropout':0.5, 'epochs':100, 'lr':5e-7, 'log_reg':False}]
# config_list = [{'embed_dim':1024, 'num_heads':64, 'num_layers':2, 'dropout':0.5, 'epochs':100, 'lr':1e-4, 'log_reg':False}]
# model_name = 'LLaVA-7B-t0'
# prompt = 'mq'
# evaluate_mad_2(model_name=model_name, prompt=prompt, tau=False, logits_used=20, trained_model_path=None, config_list=config_list)




config_list = [{'embed_dim':1024, 'num_heads':64, 'num_layers':2, 'dropout':0.5, 'epochs':100, 'lr':5e-7, 'log_reg':False}]
# config_list = [{'embed_dim':1024, 'num_heads':64, 'num_layers':2, 'dropout':0.5, 'epochs':100, 'lr':1e-4, 'log_reg':False}]

evaluate_any_dataset(
    model_name="LLaVA-7B-t0",
    dataset_name="MAD",
    prompt="mq",
    tau=False,
    logits_used=20,
    trained_model_path="./saved_models/LLaVA-7B-t0_MAD_mq_logits_used-20_embed_dim-1024_num_heads-64_num_layers-2_dropout-0.5_epochs-100_lr-5e-07_log_reg-False.pt",
    config_list=config_list
)

evaluate_any_dataset(
    model_name="LLaVA-7B-t0",
    dataset_name="MAD",
    prompt="oeh",
    tau=False,
    logits_used=20,
    trained_model_path="./saved_models/LLaVA-7B-t0_MAD_mq_logits_used-20_embed_dim-1024_num_heads-64_num_layers-2_dropout-0.5_epochs-100_lr-5e-07_log_reg-False.pt",
    config_list=config_list
)

evaluate_any_dataset(
    model_name="LLaVA-7B",
    dataset_name="POPE",
    prompt="oeh",
    tau=False,
    logits_used=20,
    trained_model_path="./saved_models/LLaVA-7B-t0_MAD_mq_logits_used-20_embed_dim-1024_num_heads-64_num_layers-2_dropout-0.5_epochs-100_lr-5e-07_log_reg-False.pt",
    config_list=config_list
)

evaluate_any_dataset(
    model_name="LLaVA-7B-2",
    dataset_name="POPE",
    prompt="mq",
    tau=False,
    logits_used=20,
    trained_model_path="./saved_models/LLaVA-7B-t0_MAD_mq_logits_used-20_embed_dim-1024_num_heads-64_num_layers-2_dropout-0.5_epochs-100_lr-5e-07_log_reg-False.pt",
    config_list=config_list
)

