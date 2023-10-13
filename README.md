# CS 598 Midterm Project 
Summary generation toolkit with configurable models.


## Introduction

Thank you for going through this course project for CS598 - Learning in Humans and Machines.
This readme file incorporates all the steps necessary to set up the application and get it up and running.

## Authors

- [@dhanur-sharma](https://github.com/dhanur-sharma)

## Requirements
Please ensure that the following dependencies are installed before proceeding with the installation:
- Ubuntu 22.04.3 LTS
- Python 3.10
- git (git version 2.39.1)
- Pip 23.2

I'd recommend running it on ilab to ensure a smoother experience so all package dependencies are resolved.

## Installation
### Step 1: Unzip the folder
After navigating to the directory you'd like to install the application in, run the following command to unzip the folder:

```bash
sudo apt-get install unzip
unzip code.zip -d code
cd code
```

You're ready to run the application.

### Step 2: Set configurations
Open the config.py file to access the parameters to change. Change the FILE_PATH to point to the data file 'hcV3-stories.csv' and the range of stories to be generated.

### Step 3: Run the program
Run the following commands on ilab to ensure that the process is complete even if the SSH connection is interrupted:

```bash
keep-job 24
nohup python3 run.py
```

The output will be generated in the outputs directory and logs will be generated in nohup.out in the same folder.

Note: The API listed will only be valid as long as there are compute credits associated with this account.
