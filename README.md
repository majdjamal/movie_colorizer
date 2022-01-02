# Movie Colorizer

Predicting color to old black-and-white movies. This module incorporates the Pix2Pix neural network architecture. The network are trained with movie frames from a color movie. These training movie frames are divided into grayscale and color. Grayscale is used in the generator, and color frames are used to validate predictions.

**Pix2Pix implementation found in model/pix2pix.py**

> **_NOTE:_**  Weights exceeds 100MB. Therefore, they are not uploaded in this repository. Send a message to majdj@kth.se to obtain them. When you get them, place the .h5 file in data/weights/

## Predict colors to a black-and-white movie frame

* Navigate to the repository

* Setup a virtual environment

```bash
python3 -m venv movie_colorizer
```

```bash
source movie_colorizer/bin/activate
```

* Install Required Utility Packages

```bash
pip3 install -r requirements.txt
```
* Place your .jpg image in data/test/Y

* Run,

```bash
python3 main.py --predict 'data/test/Y/[name_of_your_file].jpg'
```

* Navigate to data/result/frame_prediction to obtain the result
