import timm
import torch
import torch.nn as nn

            
            
class EmoticNet(nn.Module):
    def __init__(self, backone:str = 'mobinetv2', age_classes:int=1, gender_classes:int=1, emotion_classes:int=8):
        super(EmoticNet, self).__init__()
        
        if backone == 'mobinetv2':
            mobilenet_v2 = timm.create_model('mobilenetv2_100', pretrained=True) 
            self.featurenet = nn.Sequential(*list(mobilenet_v2.children())[:-3])
            encoder_ch_in = 1280
        elif backone == 'mobinetv3':
            mobilenet_v3 = timm.create_model('tf_mobilenetv3_large_100', pretrained=True)
            self.featurenet = nn.Sequential(*list(mobilenet_v3.children())[:-4])
            encoder_ch_in = 960
        else:
            raise NotImplemented
        
        opt_ch = 512
        # self.set_parameter_requires_grad(self.featurenet, False)
        
        # body neck conv
        self.block_age = self.conv_block(ch_in=encoder_ch_in, ch_out=opt_ch)
        self.adap_age = nn.AdaptiveAvgPool2d((1,1))
        self.block_gender = self.conv_block(ch_in=encoder_ch_in, ch_out=opt_ch)
        self.adap_gender = nn.AdaptiveAvgPool2d((1,1))
        self.block_emotion = self.conv_block(ch_in=encoder_ch_in, ch_out=opt_ch)
        self.adap_emotion = nn.AdaptiveAvgPool2d((1,1))
        
        # build head layers
        self.fc_age = nn.Linear(in_features=opt_ch, out_features=age_classes)
        self.fc_gender = nn.Linear(in_features=opt_ch, out_features=gender_classes)
        self.fc_emotion = nn.Linear(in_features=opt_ch, out_features=emotion_classes)
        
    def forward(self, x:torch.Tensor):
        x = self.featurenet(x)    
  
        # neck + head layer 
        x_age = self.block_age(x)
        x_age = self.adap_age(x_age)  # globalpool for first head
        x_age= x_age.view(x.shape[0], -1)
        x_age = self.fc_age(x_age)
        
        x_gender = self.block_gender(x)
        x_gender = self.adap_gender(x_gender) # globalpool for first head
        x_gender = x_gender.view(x.shape[0], -1)
        x_gender = self.fc_gender(x_gender).sigmoid()
        
        x_emotion = self.block_emotion(x)
        x_emotion = self.adap_emotion(x_emotion) # globalpool for first head
        x_emotion = x_emotion.view(x.shape[0], -1)
        x_emotion = self.fc_emotion(x_emotion)
        
        return x_age, x_gender, x_emotion
    
    def conv_block(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.Hardswish(),
            nn.Dropout(0.3, inplace=False),
        )
        
    def set_parameter_requires_grad(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad
            
            
            
if __name__ == '__main__':
    IMG_SIZE = 224
    img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    
    model = EmoticNet(backone='mobinetv2')
    model.eval()
    age, gender, emotion = model.forward(img)
    print("Age: ", age.shape)
    print("Gender: ", gender.shape)
    print("emotion: ", emotion.shape)
    

    
    