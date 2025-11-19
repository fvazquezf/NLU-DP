from conllu_reader import ConlluReader
from algorithm import ArcEager
import pickle

def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    for token in trees[0]:
        print (token)
    print ()
    return trees


"""
ALREADY IMPLEMENTED
Read and convert CoNLLU files into tree structures
"""
# Initialize the ConlluReader
reader = ConlluReader()
train_trees = read_file(reader,path="en_partut-ud-train_clean.conllu", inference=False)
dev_trees = read_file(reader,path="en_partut-ud-dev_clean.conllu", inference=False)
test_trees = read_file(reader,path="en_partut-ud-test_clean.conllu", inference=True)

"""
We remove the non-projective sentences from the training and development set,
as the Arc-Eager algorithm cannot parse non-projective sentences.

We don't remove them from test set set, because for those we only will do inference
"""
train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)

print ("Total training trees after removing non-projective sentences", len(train_trees))
print ("Total dev trees after removing non-projective sentences", len(dev_trees))


import os
import pickle
from conllu_reader import ConlluReader
from algorithm import ArcEager

def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    for token in trees[0]:
        print (token)
    print ()
    return trees


"""
ALREADY IMPLEMENTED
Read and convert CoNLLU files into tree structures
"""
# Initialize the ConlluReader
reader = ConlluReader()
train_trees = read_file(reader,path="en_partut-ud-train_clean.conllu", inference=False)
dev_trees = read_file(reader,path="en_partut-ud-dev_clean.conllu", inference=False)
test_trees = read_file(reader,path="en_partut-ud-test_clean.conllu", inference=True)

"""
We remove the non-projective sentences from the training and development set,
as the Arc-Eager algorithm cannot parse non-projective sentences.

We don't remove them from test set set, because for those we only will do inference
"""
train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)

print ("Total training trees after removing non-projective sentences", len(train_trees))
print ("Total dev trees after removing non-projective sentences", len(dev_trees))

#Create and instance of the ArcEager
arc_eager = ArcEager()
actions = {ArcEager.LA, ArcEager.RA, ArcEager.SHIFT, ArcEager.REDUCE}
dataset_path = "dataset.pkl"

if os.path.exists(dataset_path):
    print(f"\nLoading existing dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        training_samples = data["training_samples"]
        dev_samples = data["dev_samples"]
        deprels = data["deprels"]
    print("Dataset loaded successfully.")
else:
    deprels = set()
    for tree in train_trees:
        for token in tree:
            if token.dep and token.dep != "_":
                deprels.add(token.dep)

    print(f"\nPossible Actions: {sorted(list(actions))}")
    print(f"Total unique dependency labels: {len(deprels)}")

    print("\nGenerating gold-standard transitions (training samples) using Arc-Eager Oracle...")
    training_samples = []
    for i, tree in enumerate(train_trees):
        samples = arc_eager.oracle(tree) 
        training_samples.extend(samples)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(train_trees)} sentences...", end='\r')
    
    print(f"\nGeneration complete. Total training samples: {len(training_samples)}")

    print("\nGenerating development samples using Arc-Eager Oracle...")
    dev_samples = []
    for i, tree in enumerate(dev_trees):
        samples = arc_eager.oracle(tree)
        dev_samples.extend(samples)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dev_trees)} sentences...", end='\r')
    print(f"\nGeneration complete. Total dev samples: {len(dev_samples)}")

    print(f"Saving dataset to {dataset_path}...")
    with open(dataset_path, "wb") as f:
        pickle.dump({
            "training_samples": training_samples,
            "dev_samples": dev_samples,
            "deprels": deprels,
            "actions": actions
        }, f)
    print("Dataset saved.")

# Validate length against outputSample
expected_len = 81182
current_len = len(training_samples)

print(f"\nValidation: Current training samples: {current_len} | Expected: {expected_len}")
if current_len == expected_len:
    print("SUCCESS: Sample count matches the expected output.")
else:
    print(f"WARNING: Sample count mismatch! Expected {expected_len}, but got {current_len}.")



# TODO: Complete the ArcEager algorithm class.
# 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.
# 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.
# 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.

# TODO: Implement the 'state_to_feats' function in the Sample class.
# This function should convert the current parser state into a list of features for use by the neural model classifier.

# TODO: Define and implement the neural model in the 'model.py' module.
# 1. Train the model on the generated training dataset.
# 2. Evaluate the model's performance using the development dataset.
# 3. Conduct inference on the test set with the trained model.
# 4. Save the parsing results of the test set in CoNLLU format for further analysis.

# TODO: Utilize the 'postprocessor' module (already implemented).
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
# 3. Process the file: trees = postprocessor.postprocess(path)
# 4. Save the processed trees to a new output file.