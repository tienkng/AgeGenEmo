# AgeGenEmo
This repo compare the performance of CNN vs KAN network in Age, Gender and Emotion prediction in multi task learning.

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