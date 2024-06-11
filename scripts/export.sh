python src/onnx_sp/onnx_export.py \
    --checkpoint_path weights/sample-epoch=17.ckpt \
    --batch_size 1 \
    --dynamic-batch \
    --cleanup \
    --simplify