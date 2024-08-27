import cv2
import numpy as np
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

EMOTIONS = {0:'Anger', 1:'Contempt', 2:'Disgust',
            3:'Fear', 4:'Happiness', 5:'Neutral',
            6:'Sadness', 7:'Surprise'}


class EmoticONNX:
    def __init__(self, model_path:str) -> None:
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
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
    
    def inference(self, img) -> dict:
        """ Predict function
            Args:
                - img: An image source
            Return:
                Return a dictionary {
                    'age': <int> age estimation, 
                    'gender': <string> Male or Female classification, 
                    'emotion': <string> Emotion classification
                }
        """
        assert isinstance(img, np.ndarray), f'Type input model is a list, len(image) = {len(img)}, '\
                                            'should be use .batch_inference(imgs)'
        
        if len(img.shape) == 3:
            img = self.preprocess(img)
            
        results = self.model.run(
            [self.opt1_name, self.opt2_name, self.opt3_name], 
            {self.inp_name: img.astype("float32")}
        )
        if results[0].shape[0] > 1:
            return [self.postprocess([_age, _gen, _emo]) for _age, _gen, _emo in zip(*results)]
        else:
            return [self.postprocess(results)]
    
    def batch_inference(self, imgs: list[np.array]) -> list[dict]:
        tensor_imgs = []
        for x in imgs:
            tensor_imgs.append(self.preprocess(x))
            
        tensor_imgs = np.stack(tensor_imgs, axis=0).squeeze()
        
        return self.inference(tensor_imgs)
        
    
    def preprocess(self, image:np.array, img_size=(224, 224), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> np.array:
        # Resize image
        image = cv2.resize(image, img_size)
        image = image.transpose((2, 0, 1)) / 255.0  # Convert to channels first and normalize
        # Normalize image
        image = (image - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
        image = np.expand_dims(image, 0)
        
        return image
    
    def postprocess(self, result: np.array) -> dict:
        age = round(result[0].item())
        gender = 'Female' if result[1].item() > 0.5 else 'Male'
        index_emotion = self.softmax(result[2])
        
        return {'age': age, 'gender': gender, 'emotion' : EMOTIONS[index_emotion]}
               
    def softmax(self, x:np.array) -> int:
        return np.argmax(np.exp(x)/np.sum(np.exp(x)))

    
if __name__ == '__main__':
    import os
    import cv2
    
    model = EmoticONNX(
        os.path.join('weights', 'sample-epoch=17.onnx')
    )
    
    img = cv2.imread('nvt.png')
    try:
        result = model.inference([img, img])
    except:
        print("\nUsing batch inference")
        result = model.batch_inference([img, img, img, img])
    print(result)