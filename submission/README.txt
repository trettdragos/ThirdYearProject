jupyter notebooks:
    - PyTorchMnistUpscalerAutoencoder.ipynb: This code contains the process of upsampling images from the MNIST dataset.
    - prepare_video.ipynb: this a utility notebook. It is used to prepare the videos (scale them to required size)
    - MovieFramer.ipynb: this is the process of generating frames using the simple strategy of concatenating the left/right encoders and decoding to generate new frames
    - MovieFramerMark2.ipynb: this is the improved version of MovieFramer, we implement in addition the skip paths strategy to remove artifacts in the generated frames