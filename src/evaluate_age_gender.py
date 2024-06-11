import cv2
import numpy as np
import torch
import glob
import argparse
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.regression import MeanAbsoluteError

import tensorflow as tf

from utils import attem_load, preprocess
from modeling.cnn import EmoticNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
age_metric = MeanAbsoluteError().to(device)
gender_metric= BinaryAccuracy().to(device)
    


def pytorch_eval(args):
    # load model
    model = EmoticNet()
    model = attem_load(model=model, checkpoint_path=args.checkpoint_path)
    model = model.to(device)
    model.eval()

    for x in tqdm(glob.glob(f'{args.data_path}/*')):
        img = cv2.imread(x)[:,:,::-1]
        img = preprocess(img, return_tensor='pt')
        # process label
        label_age  = int(x.split('/')[-1].split('_')[0])
        label_gender  = int(x.split('/')[-1].split('_')[1])
        
        with torch.no_grad():
            pred_age, pred_gender, _ = model(img.to(device))
            pred_age = pred_age.squeeze()
            pred_gender = pred_gender.squeeze()
            
            age_metric.update(torch.tensor([pred_age]).to(device), torch.tensor([label_age]).to(device))
            gender_metric.update(torch.tensor([pred_gender]).to(device), torch.tensor([label_gender]).to(device))

    age_mae = age_metric.compute()
    gender_acc = gender_metric.compute()
    
    return age_mae, gender_acc


def tf_eval(args):
    age_model = tf.lite.Interpreter(model_path=args.tf_checkpoint)
    age_model.allocate_tensors()
    input_details = age_model.get_input_details()
    output_details = age_model.get_output_details()
    age_target_size = int(224)

    
    for x in tqdm(glob.glob(f'{args.data_path}/*')):
        # process label
        label_age  = int(x.split('/')[-1].split('_')[0])
        label_gender  = int(x.split('/')[-1].split('_')[1])
        
        img = cv2.imread(x)
        
        faces = np.empty((1, age_target_size, age_target_size, 3))
        faces[0] = cv2.resize(img, (age_target_size, age_target_size))

        faces = faces.astype(np.float32)
        age_model.set_tensor(input_details[0]["index"], faces)
        age_model.invoke()
        predicted_genders = age_model.get_tensor(output_details[0]["index"])
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = age_model.get_tensor(output_details[1]["index"]).dot(ages).flatten()
        pred_gender = 0 if predicted_genders[0][0] < 0.5 else 1
        
        age_metric.update(torch.tensor([predicted_ages[0]]).to(device), torch.tensor([label_age]).to(device))
        gender_metric.update(torch.tensor([pred_gender]).to(device), torch.tensor([label_gender]).to(device))
        
    age_mae = age_metric.compute()
    gender_acc = gender_metric.compute()
    
    return age_mae, gender_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model args
    parser.add_argument('--checkpoint_path', default=None, help='saved logging and dir path')
    parser.add_argument('--tf_checkpoint', default=None, help='saved logging and dir path')
    parser.add_argument('--data_path', default=None, help='data path')
    opt = parser.parse_args()
    
    pt_age_mae, pt_gender_acc = pytorch_eval(opt) # pytorch
    age_metric.reset()
    gender_metric.reset()
    tf_age_mae, tf_gender_acc = tf_eval(opt) # tensorflow lite
    
    print("\nNow Age mae: ", pt_age_mae)
    print("Now Gender: ", pt_gender_acc)
    print("Previou Age mae: ", tf_age_mae)
    print("Previous Gender: ", tf_gender_acc)
    print()