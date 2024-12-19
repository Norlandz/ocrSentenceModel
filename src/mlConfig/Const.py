import numpy as np

mode_ImgRgb = False
mode_UsePretrainedWeight = True

RGB_MEAN_IMAGENET = (0.485, 0.456, 0.406)
RGB_STD_IMAGENET = (0.229, 0.224, 0.225)
BW_MEAN_IMAGENET = np.average(RGB_MEAN_IMAGENET)
BW_STD_IMAGENET = np.average(RGB_STD_IMAGENET)

imageInputSize = (2**8 * 6, 2**5 * 3)

ctcBlankChar = "■"  # <bl> cause html cannot render...
vocabulary = [ctcBlankChar] + list(r"""0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ .,-()[];:!?'"`~|+=*^/\<>{}@#$%&""")

idx2char = {k: v for k, v in enumerate(vocabulary)}
char2idx = {v: k for k, v in idx2char.items()}

num_chars = len(vocabulary)

########################

rnn_hidden_size = 256
