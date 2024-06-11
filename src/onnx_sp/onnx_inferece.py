import cv2
import numpy as np
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL



class EmoticONNX:
    def __init__(self, model_path) -> None:
        self.load_model(model_path)

    def load_model(self, model_path: str):
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.inp_name = self.model.get_inputs()[0].name
        self.opt1_name = self.model.get_outputs()[0].name
        self.opt2_name = self.model.get_outputs()[1].name
        self.opt3_name = self.model.get_outputs()[2].name
        
        _, h, w, _ = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)
    
    def inference(self, img):
        if isinstance(img, list) and len(img) < 1:
            img = self.preprocess(img)
            
        result = self.model.run(
            [self.opt1_name, self.opt2_name, self.opt3_name], 
            {self.inp_name: img.astype("float32")}
        )

        return result
    
    def batch_inference(self, imgs):
        tensor_imgs = []
        for x in imgs:
            tensor_imgs.append(self.preprocess(img))
            
        tensor_imgs = np.stack(tensor_imgs, axis=0).squeeze()
        
        return self.inference(tensor_imgs)
        
    
    def preprocess(self, image, img_size=(224, 224), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        # Resize image
        image = cv2.resize(image, img_size)
        image = image.transpose((2, 0, 1)) / 255.0  # Convert to channels first and normalize
        # Normalize image
        image = (image - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
        image = np.expand_dims(image, 0)
        
        return image
    
    # def postprocess(self, result):
        

    
if __name__ == '__main__':
    import os
    import cv2
    
    model = EmoticONNX(
        os.path.join('weights', 'sample-epoch=17.onnx')
    )
    
    img = cv2.imread('nvt.png')
    result = model.batch_inference([img, img, img])
    
    print("\n", result)