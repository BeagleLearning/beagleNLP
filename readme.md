Beagle NLP
=========

Installation
------------
1. Have a working installation of python 3.6.x. If you don't, I'd suggest
installing `pyenv` and using that for managing multiple python installs.
2. Install `virtualenv`. I believe you can also use the built-in virt tool
that comes with python 3 but have not tested this.
3. Create a new virtual environment called "virt" in this repo folder by
running `virtualenv virt`
4. Activate that virtual environment with `source virt/bin/activate`
5. You now are running in a self-contained python environment. Packages
installed here will not be installed globally, and you can easily unload the
whole environment by running `deactivate` at any time.
5. Run `pip install -r requirements.txt`. This may take a while and requires a GB or two of ram for
downloading and installing the libraries of word vectors and word freqencies.

Torch and Transformers models
-----------------------------
1. Check the `expected_directory_structure.txt` file and make sure your `resources` 
directory has the same structure and content
2. Follow the links in the .txt file to download the relevant content (~1.5GB)

Use
___

1. Make sure your virtual env is active (`source virt/bin/activate`)
2. Start the server with `python application.py`


ALL CODE IN THIS PROJECT COPYRIGHT BEAGLE LEARNING, INC 2020
