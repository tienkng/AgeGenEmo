# AgeGenEmo
This repo compare the performance of CNN vs KAN network in Age, Gender and Emotion prediction in multi task learning.

## Dataset
The structure of the data
```
|__ utkface_aligned
|   |__train
|       |__ image1.jpg
|       |__ image2.jpg
|       |__ image3.jpg
|   |__test
|       |__ ...
|   |__label_train.txt
|   |__label_test.txt
```

The format of label path
```
image1.jpg	Surprise
image2.jpg	Happiness
image3.jpg	Contempt
```

## Checkpoint 
Download pretrained checkpoint [here]()

## ONNX 
For export onnx
```
python src/onnx_sp/onnx_export.py \
    --checkpoint_path weights/sample-epoch=17.ckpt \
    --batch_size 1 \
    --dynamic-batch \
    --cleanup \
    --simplify
```