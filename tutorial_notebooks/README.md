# Welcome to the Tutorial on Deep Learning for Human Activity Recognition

This is the offical GitHub page of the tutorial "Deep Learning for Human Activity Recognition" first presented at the 2021 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp’21)/ International Symposium on Wearable Computers 21' (ISWC 21'). If you are interested in going through the whole tutorial please visit our GitHub page. [[cite our work]](#cite)

## Abstract
Physical activities play a crucial role in the way we structure our lives. Which activity, and how it is performed, can reveal a person’s intention, habit, fitness, and state of mind; it is therefore not surprising that a range of research fields, from cognitive science to healthcare, display a growing interest in the machine recognition of human activities. In 2014, Bulling et al. designed and organized an exceptionally well-received tutorial on human activity recognition from wearable sensor data. They introduced concepts such as the Activity Recognition Chain (ARC), a framework for designing and evaluating activity recognition systems, as well as a case study demonstrating how to work with this ARC. Within the last decade, deep learning methods have shown to outperform classical Machine Learning algorithms. We argue that releasing an updated tutorial that is adapted to work with deep learning techniques is long overdue. This tutorial introduces the Deep Learning Activity Recognition Chain (DL-ARC), which encompasses the advances that have been made over the years within the field of deep learning for human activity recognition and deep learning. Our work directly ties into the works of Bulling et al. and functions as a step-by-step framework to apply deep learning to any activity recognition use case. Within this tutorial, we will show how state-of-the-art models can be achieved, while along the way explaining all design choices in detail. This tutorial functions as a guide in the typical processes to design and evaluate deep learning architectures, once a human activity dataset has been recorded and annotated. We show through code snippets in a step-by-step process why certain steps are needed, how they affect the system’s outcome, and which pitfalls present themselves when designing a deep learning classifier. Participants do not need prior knowledge in human activity recognition or deep learning techniques, but should be familiar with programming in Python.

## Running the notebooks

### Google Colab
To work through the tutorial, we recommend you to use Google Colab. It offers an easy way for you to quickly run the code without needing to worry about your local setup. The links to each notebook are:

- Data Collection and Analysis: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/data_collection.ipynb
- Preprocessing: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/preprocessing.ipynb
- Evaluation: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/evaluation.ipynb
- Training: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/training.ipynb
- Validation and Testing: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/validation_and_testing.ipynb
- Next Research Steps: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/next_research_steps.ipynb

Note that these notebooks will have blanks, which you need to fill in! The links to the solution files are:

- Data Collection and Analysis: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/solutions/data_collection_solution.ipynb
- Preprocessing: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/solutions/preprocessing_solution.ipynb
- Evaluation: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/solutions/evaluation_solution.ipynb
- Training: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/solutions/training_solution.ipynb
- Validation and Testing: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/solutions/validation_and_testing_solution.ipynb
- Next Research Steps: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/solutions/next_research_steps_solution.ipynb
- (Optional) Example Point Classification 2D: https://colab.research.google.com/github/mariusbock/dl-for-har/blob/main/tutorial_notebooks/solutions/example_point_classification2d.ipynb


**Important**: In order for the notebooks to be run properly, one needs to run some mandatory code (seen in the first code cell of each notebook). Set `use_colab=True` in order for the code to run every necessary prerequisite when using Google Colab.

### Local deployment

All notebooks are also possible to be run locally. To do so make sure that your Python distribution has all necessary packages installed which are mentioned in the `requirements.txt` of the main GitHub repository. Also make sure to set `use_colab=False` within the first code cell of each notebook. Using local deployment you will be also able to switch out the dataset which is loaded within the tutorial notebooks. To do so download the preprocessed datasets which work with the repository from [here](https://uni-siegen.sciebo.de/s/sMWQ2vJhDzM6sil) (PW: iswc21).

## Citation
```
  @article{bock2021tutorial,
    author = {Bock, Marius and Hölzemann, Alexander and Moeller, Michael and Van Laerhoven, Kristof},
    title = {Tutorial on Deep Learning for Human Activity Recognition},
    year = {2021},
    journal = {CoRR},
    volume = {abs/2110.06663},
    doi = {10.48550/arXiv.2110.06663},
    url = {https://arxiv.org/abs/2110.06663}
}
```
