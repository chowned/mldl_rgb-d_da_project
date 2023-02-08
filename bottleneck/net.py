import torch.nn as nn
from torchvision import models

#import pretrainedmodels
#from data_loader import INPUT_RESOLUTION

#INPUT_LAYER = 64

#nn.Dropout(p=0.5,inplace=True)

class ResBase(nn.Module):
    def __init__(self):
        super(ResBase, self).__init__()
        # Initialize pre-trained resnet34
        model_resnet = models.resnet18(pretrained=True)

        # "Steal" pretrained layers from the torchvision pretrained Resnet18
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        #self.bn1 = nn.BatchNorm2d(3)
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool

        input_l=64
        l1 = 50
        l2 = 40
        l3 = 34
        l4 = 24
        l5 = 14
        l6 = 24
        l7 = 34
        l8 = 40
        l9 = 50
        kernel=3

        self.encoder = nn.Sequential(
            #implementing a neuron bottleneck



            nn.Conv2d(in_channels=input_l, out_channels=l1, kernel_size=kernel, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(l1),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l1, out_channels=l2, kernel_size=kernel, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(l2),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l2, out_channels=l3, kernel_size=kernel, stride=1, padding=1),
            nn.BatchNorm2d(l3),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l3, out_channels=l4, kernel_size=kernel, stride=1, padding=1),
            nn.BatchNorm2d(l4),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l4, out_channels=l5, kernel_size=kernel, stride=1, padding=1),
            nn.BatchNorm2d(l5),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l5, out_channels=l6, kernel_size=kernel, stride=1, padding=1),
            nn.BatchNorm2d(l6),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l6, out_channels=l7, kernel_size=kernel, stride=1, padding=1),
            nn.BatchNorm2d(l7),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l7, out_channels=l8, kernel_size=kernel, stride=1, padding=1),
            nn.BatchNorm2d(l8),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l8, out_channels=l9, kernel_size=kernel, stride=1, padding=1),
            nn.BatchNorm2d(l9),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=l9, out_channels=input_l, kernel_size=kernel, stride=1, padding=1),
            nn.BatchNorm2d(input_l),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            )

        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2 #after is 128

        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4


        self.avgpool = model_resnet.avgpool

    def forward(self, x):

        x = self.conv1(x)
        x = self.encoder(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        #x = self.encoder(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Non-pooled tensor
        x_p = x
        x = self.avgpool(x)
        # Flatten the pooled tensor
        x = x.flatten(start_dim=1)

        # Return both
        return x, x_p


class AUTOENCODER(nn.Module):
    def __init__(self):
        super(ResBase,self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        #self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)





        self.encoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(p=0.5,inplace=True),
            nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 16, 3, stride=2, padding=1),
        nn.ReLU(True),
        nn.Dropout(p=0.5,inplace=True),
        nn.MaxPool2d(2, stride=1),
        nn.Conv2d(16, 8, 3, stride=3, padding=1),
        nn.ReLU(True),
        nn.Dropout(p=0.5,inplace=True),
        nn.MaxPool2d(2, stride=2),
        )
        self.decoder = nn.Sequential(
       nn.ConvTranspose2d(8, 16, 3, stride=2),
       nn.ReLU(True),
       nn.Dropout(p=0.5,inplace=True),
       nn.ConvTranspose2d(16, 32, 5, stride=3, padding=1),
       nn.ReLU(True),
       nn.Dropout(p=0.5,inplace=True),
       nn.ConvTranspose2d(32, 256, 2, stride=2, padding=1),
       nn.Dropout(p=0.5,inplace=True),
       nn.Tanh()
        )

        self.layer4 = model_resnet.layer4

        self.avgpool = model_resnet.avgpool




    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.encoder(x)
        x = self.decoder(x)
        x = self.layer4(x)
        # Non-pooled tensor
        x_p = x
        x = self.avgpool(x)
        # Flatten the pooled tensor
        x = x.flatten(start_dim=1)

        # Return both
        return x, x_p


class ResClassifier(nn.Module):
    def __init__(self, input_dim=1024, class_num=47, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.fc2 = nn.Linear(1000, class_num)
        self.dropout_p = dropout_p

    def forward(self, x):
        emb = self.fc1(x)
        logit = self.fc2(emb)

        return logit


class RelativeRotationClassifier(nn.Module):
    def __init__(self, input_dim, projection_dim=100, class_num=4):
        super(RelativeRotationClassifier, self).__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(self.input_dim, self.projection_dim, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(self.projection_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(self.projection_dim, self.projection_dim, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(self.projection_dim),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.projection_dim * 3 * 3, self.projection_dim),
            nn.BatchNorm1d(self.projection_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(projection_dim, class_num)

    def forward(self, x):
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
