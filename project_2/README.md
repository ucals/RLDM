Installing in a clean Ubuntu image:
- sudo apt install python3-pip python python-setuptools python-dev python-augeas gcc swig dialog
- pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose gym torch torchvision
- Install http://www.kiranjose.in/blogs/getting-started-with-openai-gym-part-1-installation-and-configuration/:
    - git clone https://github.com/pybox2d/pybox2d
    - cd pybox2d/
    - python3 setup.py clean
    - python3 setup.py build
    - sudo python3 setup.py install
    - ssh-keygen -t rsa -b 4096 -C "souza@gatech.edu"
    
# Experiments

1- Different learners
- First version done
- TODO: run another version, where agents keep learning for longer, and then test for 100 episodes (1b)

2- Different learning rates
- First version done
- TODO: run another version, 1 run/experiment, but with more learning rates (scatter plot), and for different optimizers (2b)

3- Different Gammas
- First version taking too long
- TODO: run a simpler version, 5 runs/experiment (3b)

4- Different Exploration strategies
- TODO: test different epsilon curves

Next steps:
- Deploy e3b, simpler Gamma experiment, in AWS (DONE)
- Parametrize different exploration strategies in experiment protocol (DONE)
- Deploy e4, different exploration strategies, in AWS (DONE)
- Do small change in code to execute e1b and deploy in AWS
- Prepare code to do single-run experiments in parallel
- Parametrize optimizer in experiment protocol
- Deploy 2b, different learning rates vs. optimizer., in AWS

