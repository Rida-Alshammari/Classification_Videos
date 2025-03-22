# **Classification Violent, Non-violent Videos**

![wallpaper](https://fiverr-res.cloudinary.com/images/q_auto,f_auto/gigs/217475863/original/24da1da041b6bbf8e209eab8841e8a7855b659e2/code-an-advanced-python-flask-website.jpg)

## Project Description

In this project, I have trained a model to classify violent, non-violent actions in videos. The model is trained on a dataset of videos, and the model is then used to classify new videos. The model is trained on a dataset of videos, and the model is then used to classify new videos. The model is built using [Flask](https://flask.palletsprojects.com/en/2.0.x/), [TensorFlow](https://www.tensorflow.org/), and [JavaScript](https://www.javascript.com/). The dataset was downloaded from [Kaggle](https://www.kaggle.com/).

---

## Project Structure

The project is structured as follows:

- `dataset`:
  - violent/
  - non-violent/
- `static`:

  - images/
  - css/
    - index.css
    - result.css
  - js/
    - app.js

- `templates`:
  - index.html
  - result.html
- `uploads/`:
- `requirements.txt`:
- `train.py`:
- `app.py`:
- `model.keras`:

---

## Installation

To install the project, run the following command:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Run `python train.py` to train the model.

```bash
python train.py
```

2. Run `python app.py` to run the web app.

```bash
python app.py
```

## Requirements

The following packages are required to run the project:

- `flask>=3.0.2`
- `numpy>=1.26.4,<2.0.0`
- `opencv-python>=4.10.0.84`
- `tensorflow>=2.15.0`
- `scikit-learn>=1.4.2`

| Section | Description |
| --- | --- |
| **Dataset** | The dataset is a collection of videos of violent and non-violent actions. It is organized into two folders: one for violent actions and another for non-violent actions. |
| **Model** | The model is a convolutions neural network trained on the dataset to classify new videos as either violent or non-violent. |
| **Web App** | The web app provides a simple interface for users to upload a video and receive a classification result. |
| **Training** | The model undergoes training using the video dataset, enabling it to classify new videos accurately. |

## Technologies Used

The project is built using the following technologies:

- **Programming Language**: Python
- **Frameworks**: Flask
- **Libraries**: Tensorflow, OpenCV, NumPy, Scikit-learn
- **Markup Language**: HTML
- **Style Sheet Language**: CSS
- **Scripting Language**: JavaScript

---

## Resources and Links dataset and model

The dataset and model can be downloaded from the following links:

- [Download Link Kaggle Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
