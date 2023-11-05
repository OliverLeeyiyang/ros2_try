import rclpy
from rclpy.node import Node
import os
import torch
import torch.nn as nn
from nn_integration.model import Encoder, Decoder, ConvAutoencoder
import __main__
setattr(__main__, "ConvAutoencoder", ConvAutoencoder)
setattr(__main__, "Encoder", Encoder)
setattr(__main__, "Decoder", Decoder)

class CONV(Node):

    def __init__(self):
        super().__init__('conv_node')

        path = os.path.abspath('ros2_try/src/nn_integration')
        self.model_path = path + '/CNN_200.pth'
        print(self.model_path)
        enc = Encoder()
        dec = Decoder()
        self.net = ConvAutoencoder(enc, dec)
        self.net = torch.load(self.model_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.eval()


def main(args=None):
    rclpy.init(args=args)
    
    conv = CONV()
    
    # rclpy.spin(conv)
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()