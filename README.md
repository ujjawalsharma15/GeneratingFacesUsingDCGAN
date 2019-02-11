# GeneratingFacesUsingDCGAN
Implementation of DCGAN on celeb faces dataset in pytorch.

link to paper [here](https://arxiv.org/abs/1511.06434)

## Dataset

Dataset used is celeb faces dataset containing 2,02,599 images of celeb faces.

link to dataset [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)

## Network Architecture

Generator consists of 5 ConvTranspose layers along with BatchNorm and leaky ReLU layers, with noise z as input and the output image is of 64x64x3 in shape.

here is the image of Generator:-
![alt text](https://cdn-images-1.medium.com/max/2600/1*bAJvLTJCWstTSiV49nyzXg.png)

Discriminator consists of 5 convolutional layers and outputs the probability of being real image, along with LeakyReLU and BatchNorm layers.

here is the image of the Discriminator:-

![alt text](https://user-images.githubusercontent.com/37034031/43060075-47f274d0-8e8a-11e8-88ff-3211385c7544.png)

## Implementation Details

Used Binary Cross Entropy loss, trained Discriminator for classifying real inputs first and then on fake images seperately and adding up the gradients to subtract from the weights for gradient descent, used SGD with Momentum to train Discriminator.
For Generator used Adam optimizer, learning rate for both generator and discriminator training was set to 0.00015.

Refrence used for tricks to achieve stable training of GAN is [ganhacks](https://github.com/soumith/ganhacks)

For tracking running losses as well as to track how well our generator is performing, used [TensorboardX](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html) for visualization.

## Results

images generated during training:-

![screenshot from 2019-02-11 17-58-17](https://user-images.githubusercontent.com/33577587/52563667-1fe23a00-2e28-11e9-921f-b9438bbcab60.png)
![screenshot from 2019-02-11 17-58-27](https://user-images.githubusercontent.com/33577587/52563698-2ec8ec80-2e28-11e9-81f1-312a176305bf.png)
![screenshot from 2019-02-11 17-58-50](https://user-images.githubusercontent.com/33577587/52563748-4d2ee800-2e28-11e9-9eea-2bd19880c469.png)

images generated after training:-

![screenshot from 2019-02-11 17-59-16](https://user-images.githubusercontent.com/33577587/52563773-5cae3100-2e28-11e9-886b-c42ef80eaaf1.png)
![screenshot from 2019-02-11 17-56-41](https://user-images.githubusercontent.com/33577587/52563603-e90c2400-2e27-11e9-94e1-d0f70285a0a0.png)

Discriminator Loss:-

![screenshot from 2019-02-11 18-12-43](https://user-images.githubusercontent.com/33577587/52563939-bdd60480-2e28-11e9-9ba4-cd1eeb84fd36.png)

Generator Loss:-

![screenshot from 2019-02-11 18-12-53](https://user-images.githubusercontent.com/33577587/52563969-d8a87900-2e28-11e9-91dc-aed8376cdea7.png)
