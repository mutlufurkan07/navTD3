import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, frame_number):
        super(CNNBlock, self).__init__()
        self.activation = nn.LeakyReLU()
        # cnn part 72 x 128
        self.conv_layer1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2))  # padding same
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))  # padding valid
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))  # padding valid
        print(f"CNN parameter count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        # 1152 features

    def forward(self, depth):
        ll = depth[:, :, :, 0:128]
        ff = depth[:, :, :, 128:256]
        rr = depth[:, :, :, 256:]
        klk = torch.cat((ll, ff, rr), dim=1)
        klk = klk / 255.0
        # Layer 1
        o1 = self.conv_layer1(klk)
        o1 = self.activation(o1)

        # Layer 2
        o1 = self.conv_layer2(o1)
        o1 = self.activation(o1)

        # Layer 3
        o1 = self.conv_layer3(o1)
        o1 = self.activation(o1)

        # Layer 4
        o1 = self.conv_layer4(o1)
        o1 = self.activation(o1)

        # Layer 5
        o1 = self.conv_layer5(o1)
        o1 = self.activation(o1)
        out_features = torch.flatten(o1, 1)
        return out_features


class CNNBlock_Sketch(nn.Module):
    def __init__(self, frame_number):
        super(CNNBlock_Sketch, self).__init__()
        self.activation = nn.LeakyReLU()
        # cnn part 72 x 128
        self.conv_layer1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(4, 4), stride=(2, 2))
        self.conv_layer2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 4), stride=(2, 2))
        self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(2, 2))
        self.conv_layer4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        print(f"CNN parameter count (Sketch): {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        # 1152 features

    def forward(self, depth):
        depth = depth / 255.0
        # Layer 1
        o1 = self.conv_layer1(depth)
        o1 = self.activation(o1)

        # Layer 2
        o1 = self.conv_layer2(o1)
        o1 = self.activation(o1)

        # Layer 3
        o1 = self.conv_layer3(o1)
        o1 = self.activation(o1)

        # Layer 4
        o1 = self.conv_layer4(o1)
        o1 = self.activation(o1)

        out_features = torch.flatten(o1, 1)
        return out_features


class ActorNetwork(nn.Module):
    def __init__(self, frame_number, camera_dim, state_dims, past_steering_dims, layer1_dims, layer2_dims, action_dim):
        super(ActorNetwork, self).__init__()
        self.action_before_tanh = [0.0, 0.0]
        self.activation = nn.LeakyReLU()
        self.out_activation = nn.Tanh()
        # cnn part 72 x 128
        self.feature_extractor = CNNBlock(frame_number)
        self.feature_extractor_sketch = CNNBlock_Sketch(frame_number)
        # self.feature_extractor = ResNet(ResidualBlock, [1, 2, 1])

        with torch.no_grad():
            dummy_tensor = torch.zeros((1, *(camera_dim[0], camera_dim[1], camera_dim[2])), dtype=torch.float32)
            dummy_tensor_2 = torch.zeros((1, 4, 80, 80), dtype=torch.float32)
            out = self.feature_extractor.forward(dummy_tensor)
            out2 = self.feature_extractor_sketch.forward(dummy_tensor_2)
        self.feature_number = out.shape[1]
        feature_number_sk = out2.shape[1]
        print(f"Actor CNN Feature Num: {self.feature_number}")

        self.first = nn.Linear(self.feature_number + feature_number_sk + state_dims + past_steering_dims, layer1_dims)
        self.second = nn.Linear(layer1_dims, layer2_dims)
        self.third = nn.Linear(layer2_dims, action_dim)
        # self.third.weight.data = self.third.weight.data / 100

        print(f"actor parameter count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, sketch, depth, state, past_steer):
        # Layer 1
        o_flatten = self.feature_extractor(depth)
        sk_flatten = self.feature_extractor_sketch(sketch)
        o1 = torch.cat((o_flatten, sk_flatten, state, past_steer), dim=1)

        o1 = self.first(o1)
        o1 = self.activation(o1)

        o1 = self.second(o1)
        o1 = self.activation(o1)

        before_tanh = self.third(o1)
        if o1.shape[0] == 1:
            self.action_before_tanh[0], self.action_before_tanh[1] = before_tanh[0, 0], before_tanh[0, 1]
        o1 = self.out_activation(before_tanh)

        return o1, before_tanh

    def grad_check_hook(self, module, grad_in, grad_out):
        pass
        # print(module)
        # print(grad_in)
        # print(grad_out)


class CriticNetwork(nn.Module):
    def __init__(self, frame_dim, camera_dim, state_dim, past_steering_dims, action_dims, layer1_dims, layer2_dims):
        super(CriticNetwork, self).__init__()
        self.activation = nn.LeakyReLU()

        self.feature_extractor = CNNBlock(frame_dim)
        self.feature_extractor_sketch = CNNBlock_Sketch(frame_dim)
        # self.feature_extractor = ResNet(ResidualBlock, [1, 2, 1])

        with torch.no_grad():
            dummy_tensor = torch.zeros((1, *(camera_dim[0], camera_dim[1], camera_dim[2])), dtype=torch.float32)
            dummy_tensor_2 = torch.zeros((1, 4, 80, 80), dtype=torch.float32)
            out = self.feature_extractor.forward(dummy_tensor)
            out2 = self.feature_extractor_sketch.forward(dummy_tensor_2)
        self.feature_number = out.shape[1]
        feature_number_sk = out2.shape[1]
        print(f"Critic CNN Feature Num: {self.feature_number}")

        self.first = nn.Linear(self.feature_number + feature_number_sk + state_dim + action_dims + past_steering_dims,
                               layer1_dims)
        self.second = nn.Linear(layer1_dims + action_dims, layer2_dims)
        self.third = nn.Linear(layer2_dims + action_dims, 1)
        # with torch.no_grad():
        #    self.third.weight[0, 256] = 0.6
        #    self.third.weight[0, 257] = -1
        print(f"critic parameter count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, sketch, depth, state, action, past_steer):
        o_flatten = self.feature_extractor(depth)
        sk_flatten = self.feature_extractor_sketch(sketch)
        out = self.first(torch.cat((o_flatten, sk_flatten, state, action, past_steer), dim=1))
        out = self.activation(out)

        out = self.second(torch.cat((out, action), dim=1))
        out = self.activation(out)

        out = self.third(torch.cat((out, action[:, 0].unsqueeze(1), torch.abs(action[:, 1]).unsqueeze(1)), dim=1))
        return out
