python src/evaluate_age_gender.py  \
    --checkpoint_path 'output/sample-epoch=17.ckpt' \
    --tf_checkpoint 'trained_models/age_models/MobileNetV2_224_weights.11-3.35.tflite' \
    --data_path '/home/tiennv/nvtien/projects/AgeGenEmo/datahub/utkface_aligned_cropped_emotion/test'


python src/evaluate_emotion.py  \
    --checkpoint_path 'output/sample-epoch=17.ckpt' \
    --tf_checkpoint 'trained_models/emotion_models/0531_1949_mini_XCEPTION3.62-0.65.hdf5' \
    --data_path '/home/tiennv/nvtien/projects/AgeGenEmo/datahub/utkface_aligned_cropped_emotion/test' \
    --data_emotion_path datahub/utkface_aligned_cropped_emotion/emotion_test.txt \