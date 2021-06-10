# facial-attributes

## Installation
The installation exists of two parts. One part is setting up the dataset,
the other part is setting up the python virtual environment.

Setting up the dataset is quite straightforward. The dataset can be
downloaded from [Kaggle](https://www.kaggle.com/davidjfisher/illinois-doc-labeled-faces-dataset). 
You need to have a Kaggle account in order to download the dataset.
Once the dataset has downloaded, create a new directory in the git
repository called `data` and extract the files from the dataset in here.
To check whether things are set up correctly, try executing the following
command: `ls -la git_repository/data/front/front/A00147.jpg`. It should
return the file permissions.

The second part is a bit less straightforward to set up. As the `torch`
and `torchvision` packages have to be installed separately. See the end
of this section for instructions on how to do this.


Instructions for `virtualenv` are provided below. It should also be possible
to use `anaconda` but this has not been tested.

Create a python virtual environment and install the packages listed in
`requirements.txt`. Note that these requirements will be updated in the
future, so if something does not work, please check whether your virtual
environment is up to date. To create a python virtual environment and
install the required packages execute the following commands from the
main directory of this repository:
```
python3 -m virtualenv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

While setting this up, I used Python 3.7.3. You can find your version by
executing: `python3 --version` (or just `python --version` depending on
your OS). Generally, any version >= 3.5 should work.

If you do not have the virtual environment package installed, this can be
done as follows for any Debian derivative (i.e. Ubuntu, Mint):
```
apt install python3-virtualenv
```
or via pip:
```
pip3 install virtualenv
```

If you want to run the network on a GPU, you should run the following:
```
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

If you only want/can run the network on a CPU, installing the following
should suffice:
```
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Running the project

The whole project can be run from the Jupyter notebook: 
`FacialAttributes.ipynb`. To do so, execute:
```
source .venv/bin/activate
jupyter notebook
```
and navigate to the notebook.
