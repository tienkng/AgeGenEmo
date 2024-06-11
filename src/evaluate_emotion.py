import cv2
import numpy as np
import torch
import glob
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from keras.models import load_model

from utils import attem_load, preprocess
from modeling.cnn import EmoticNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
emotion_labels = {'Anger' : 0, 'Contempt' : 1, 'Disgust' : 2,
                'Fear' : 3, 'Happiness' : 4, 'Neutral' : 5,
                'Sadness' : 6, 'Surprise' : 7}

tf_emotion_labels = {0: 'Anger', 1: 'Dislike', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral'}

def get_list_labels(datapath):
    with open(datapath, 'r') as file:
        data = file.readlines()
    
    emotion = dict()    
    for x in data:
        x1, x2 = x.strip().split('\t')
        emotion.update({x1:x2})
        
    return emotion
        
        
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x



def pytorch_eval(args):
    # load model
    model = EmoticNet()
    model = attem_load(model=model, checkpoint_path=args.checkpoint_path)
    model = model.to(device)
    model.eval()

    label_emotions, pred_emotions = [], []
    
    for x in tqdm(glob.glob(f'{args.data_path}/*')):
        img = cv2.imread(x)[:,:,::-1]
        img = preprocess(img, return_tensor='pt')
        
        # process label
        name = x.split('/')[-1]
        list_emtion = get_list_labels(args.data_emotion_path)
        emo_name = list_emtion[name]
        label_emotions.append(emotion_labels[emo_name])
        
        with torch.no_grad():
            _, _, pred_emotion = model(img.to(device))
            pred_idx = torch.argmax(pred_emotion, dim=1).item()
        pred_emotions.append(pred_idx)
    
    return pred_emotions, label_emotions


def tf_eval(args):
    emotion_model = load_model(args.tf_checkpoint, compile=False)
    emotion_target_size = emotion_model.input_shape[1:3]

    pred_emotions, label_emotions = [], []
    for x in tqdm(glob.glob(f'{args.data_path}/*')):
        # process label
        name = x.split('/')[-1]
        list_emtion = get_list_labels(args.data_emotion_path)
        emo_name = list_emtion[name]
 
        if emo_name == 'Contempt':
            continue
        
        if emo_name == 'Anger':
            label_emotions.append(0)
        elif emo_name == 'Disgust':
            label_emotions.append(1)
        elif emo_name == 'Fear':
            label_emotions.append(2)
        elif emo_name == 'Happiness':
            label_emotions.append(3)
        elif emo_name == 'Sadness':
            label_emotions.append(4) 
        elif emo_name == 'Surprise':
            label_emotions.append(5)
        elif emo_name == 'Neutral':
            label_emotions.append(6)
        
        # Process image
        img = cv2.imread(x)
        gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (emotion_target_size))
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        emotion_results = emotion_model.predict(gray_face)
        pred_emotions.append(np.argmax(emotion_results))
    
    return pred_emotions, label_emotions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model args
    parser.add_argument('--checkpoint_path', default=None, help='saved logging and dir path')
    parser.add_argument('--tf_checkpoint', default=None, help='saved logging and dir path')
    parser.add_argument('--data_path', default=None, help='data path')
    parser.add_argument('--data_emotion_path', default=None, help='data path')
    opt = parser.parse_args()
    
    # pred_emotions, label_emotions = pytorch_eval(opt) # pytorch
    # report = classification_report(label_emotions, pred_emotions, output_dict=True)
    # macro_f1 = report['macro avg']['f1-score']
    # accuracy = report['accuracy']
    
    # confusion_matrix = confusion_matrix(label_emotions, pred_emotions)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=list(emotion_labels.keys()))
    # disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    # plt.title(f"Our model with Acc: {accuracy:.4}")
    # plt.tight_layout()
    # plt.savefig("new.png", pad_inches=5)
    
    pred_emotions, label_emotions = tf_eval(opt)
    assert len(pred_emotions) == len(label_emotions)
    report = classification_report(label_emotions, pred_emotions, output_dict=True)
    accuracy = report['accuracy']

    confusion_matrix = confusion_matrix(label_emotions, pred_emotions)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=list(tf_emotion_labels.values()))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Emotics: emotion (old) - Acc: {accuracy:.4}")
    plt.tight_layout()
    plt.savefig("old.png", pad_inches=5)