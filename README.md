# Movie Colorizer

This model contains a deep learning model that adds color to black-and-white movies. The predictor builds on the Pix2Pix neural network architecture.

![ Alt text](data/result_sample.gif)

(Figure 1. Color predictions for black-and-white movie frames. Predictions were yellowish in general, but some structures were astonishing. The movie in context is Kansas City Confidential, released in 1952.)

**Pix2Pix implementation found in model/pix2pix.py**


## Predict colors to a black-and-white movie frame

> **_NOTE:_**  Weights exceeds 100MB. Therefore, they are not uploaded in this repository. Send a message to majdj@kth.se to obtain them. When you get them, place the .h5 file in data/weights/

* Navigate to the repository

* Place your .jpg image in data/test/Y

* Run,

```bash
python3 main.py --predict --test_path 'data/test/Y/[name_of_your_file].jpg'
```

* Navigate to data/result/frame_prediction to obtain the result
