Installing in a clean Ubuntu image:
- sudo apt install python3-pip
- pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose gym torch torchvision
- Install http://www.kiranjose.in/blogs/getting-started-with-openai-gym-part-1-installation-and-configuration/:
    - sudo apt-get install python python-setuptools python-dev python-augeas gcc swig dialog
    - git clone https://github.com/pybox2d/pybox2d
    - cd pybox2d/
    - python3 setup.py clean
    - python3 setup.py build
    - sudo python3 setup.py install