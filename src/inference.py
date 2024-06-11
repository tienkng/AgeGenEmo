import cv2
import torch
import argparse
from modeling.emotic import EmoticNet
from utils import preprocess, attem_load


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
emotion_labels = {'Anger' : 0, 'Contempt' : 1, 'Disgust' : 2,
                'Fear' : 3, 'Happiness' : 4, 'Neutral' : 5,
                'Sadness' : 6, 'Surprise' : 7}

    

def main(args):
    model = EmoticNet()
    model = attem_load(model=model, checkpoint_path=args.checkpoint_path)
    model = model.to(device)
    model.eval()
    
    img = cv2.imread(args.source)
    img = preprocess(img)
    
    with torch.no_grad():
        pred_age, pred_gender, pred_emotion = model(img.to(device))
    
    pred_age, pred_gender = pred_age.squeeze(), pred_gender.squeeze()
    pred_idx = torch.argmax(pred_emotion, dim=1).item()
    emotion =  list(emotion_labels.keys())[pred_idx]

    gender = 'Male' if pred_gender.item() < 0.5 else 'Female'
    print(f"Predict Age: {pred_age.item():.1f}", "\tGender: ", gender, "\tEmotion: ", emotion)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=None, help='checkpoint folder path')
    parser.add_argument('--source', default=None, help='image path')
    opt = parser.parse_args()
    
    main(opt)