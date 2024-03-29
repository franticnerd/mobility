#!/bin/zsh

# parameter file
para_file='./ny40k.yaml'
java_dir='../hmm-java/'
python_dir='../python-mobility/'
jar_file=$java_dir'hmm-java.jar'

# --------------------------------------------------------------------------------
# Step 1: preprocessing.
# --------------------------------------------------------------------------------

function pre {
  # python $python_dir'preprocess.py' $para_file
}

# --------------------------------------------------------------------------------
# Step 2: run the algorithms.
# --------------------------------------------------------------------------------
function run {
  java -jar -Xmx5G $jar_file $para_file
}


# --------------------------------------------------------------------------------
# Step 3: post-processing
# --------------------------------------------------------------------------------

function post {
  # python $python_dir'plot.py' $para_file
}

pre
run
post
