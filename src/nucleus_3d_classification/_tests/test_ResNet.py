from nucleus_3d_classification.models.ResNet import ResNet, Block
import torch

def test_ResNet():
    created_ResNet = ResNet(
                layers=[1,1,1,1],
                num_classes=2,
                block=Block,
                image_channels=1, ceil_mode=True,
                padding_layer_sizes=(20,21,6,6,6,7),
                leaky=True)
    
    assert created_ResNet is not None
    print("Test ResNet creation passed.")
    assert created_ResNet.padding_layer_sizes == (20,21,6,6,6,7)
    print("Test padding_layer_sizes passed.")

    # Test forward pass
    x = torch.randn(2,1,35,331,216)
    y = created_ResNet(x)
    assert y is not None
    assert y.shape == (2,2)
    print("Test forward pass passed.")

def main():
    test_ResNet()
    print("All tests passed.")

if __name__ == "__main__":
    main()

#PYTHONPATH=./src python src/nucleus_3d_classification/_tests/test_ResNet.py
