import os
import sys
import pathlib
import subprocess
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from module.neuralnetwork.MLP import MLP
from module.reader.Reader import Reader

def learn_using_single_series(mlp, no_of_last_samples, learning_data):
    for i in range(no_of_last_samples, len(learning_data) + 1):
        inputs = [data[0] for data in learning_data[i - no_of_last_samples:i]] \
                 + [data[1] for data in learning_data[i - no_of_last_samples:i]]
        outputs = [learning_data[i - 1][2], learning_data[i - 1][3]]
        mlp.learn(inputs, outputs, 0.05)


def test_using_single_series(mlp, no_of_last_samples, testing_data):
    mlp_output = []
    for i in range(no_of_last_samples, len(testing_data) + 1):
        inputs = [data[0] for data in testing_data[i - no_of_last_samples:i]] \
                 + [data[1] for data in testing_data[i - no_of_last_samples:i]]
        mlp_output.append(mlp.test(inputs))

    return mlp_output


def calculate_error(testing_data, mlp_output):
    error = []
    for i in range(len(testing_data) - len(mlp_output), len(testing_data)):
        testing_data_row = testing_data[i]
        mlp_output_row = mlp_output[i - (len(testing_data) - len(mlp_output))]
        error.append((((testing_data_row[2] - mlp_output_row[0]) ** 2) + 
                ((testing_data_row[3] - mlp_output_row[1]) ** 2)) ** 0.5)
    return error

def calculate_distribution(testing_data, mlp_output):
    errors = calculate_error(testing_data, mlp_output)
    distribution = []
    error = 0
    while len(distribution) == 0 or distribution[-1] < len(errors):
        distribution.append(len(list(filter(lambda e: e < error, errors))))
        error += 1
    distribution = [x / len(errors) for x in distribution]
    return distribution


def make_animation(testing_data, mlp_output):
    plt.clf()
    plt.plot([row[0] for row in testing_data], [row[1] for row in testing_data], "ro")
    plt.plot([row[2] for row in testing_data], [row[3] for row in testing_data], "go")
    plt.plot([row[0] for row in mlp_output], [row[1] for row in mlp_output], "bo")
    plt.draw()
    plt.pause(0.01)

def main() -> None:
    print("Reading data...")
    reader = Reader()
    learning_data_series = reader.load_learning_data_series()
    testing_data = reader.load_testing_data()

    print("Creating list of experiments...")
    experiments = []
    i = 1
    while i < len(sys.argv):
        no_of_last_samples = int(sys.argv[i])
        hidden_layers = [sys.argv[i + 2 + j] for j in range(int(sys.argv[i + 1]))]
        experiments.append([no_of_last_samples] + hidden_layers)
        i += len(hidden_layers) + 2
    print(experiments)
    
    experiment_number = 1
    for experiment in experiments:
        print("\nExperiment: " + str(experiment_number) + "/" + str(len(experiments)))
        experiment_number += 1

        print("\tCreating MLP...")
        no_of_last_samples = experiment[0]
        mlp = MLP([str(no_of_last_samples * 2)] + experiment[1:] + ['2s'])

        print("\tTraining...")
        same_error_counter = 0
        errors = [0.0]
        while same_error_counter < 10:
            #learn - single epoch
            for learning_data in learning_data_series:
                learn_using_single_series(mlp, no_of_last_samples, learning_data)

            #calculate and print error
            mlp_output = test_using_single_series(mlp, no_of_last_samples, testing_data)
            errors.append(sum(calculate_error(testing_data, mlp_output)))
            if abs(errors[-1] - errors[-2]) < 1:
                same_error_counter += 1
            else:
                same_error_counter = 0
            print("\t\t" + str(len(errors) - 1) + ": " + str(errors[-1]))

            #make animation (if there is only one experiment on the list)
            #if len(experiments) == 1:
            #    make_animation(testing_data, mlp_output)

        filename = ""
        for i in experiment:
            filename += str(i) + "_"

        print("\tSaving graph to file...")
        plt.clf()
        plt.plot([row[0] for row in testing_data], [row[1] for row in testing_data], "ro")
        plt.plot([row[2] for row in testing_data], [row[3] for row in testing_data], "go")
        plt.plot([row[0] for row in mlp_output], [row[1] for row in mlp_output], "bo")
        plt.savefig(filename + "graph.jpg")

        print("\tSaving global errors from all iterations to file...")
        errors_file = open(filename + "errors", "w")
        errors_file.writelines([str(error) + "\n" for error in errors[1:]])
        errors_file.close()

        print("\tSaving neurons' weights to file...")
        weights_file = open(filename + "weights", "w")
        weights_file.write(mlp.weights_to_string())
        weights_file.close()

        print("\tCalculating cumulative distribution and saving it to file...")
        denormalized_testing_data = reader.denormalize_testing_data(testing_data)
        denormalized_mlp_output = reader.denormalize_mlp_output(mlp_output)
        distribution_learned = calculate_distribution(denormalized_testing_data, denormalized_mlp_output)
        distribution_original = calculate_distribution(denormalized_testing_data, denormalized_testing_data)
        plt.clf()
        plt.plot(range(len(distribution_learned)), distribution_learned, "ro")
        plt.plot(range(len(distribution_original)), distribution_original, "bo")
        plt.savefig(filename + "distribution_graph.jpg")
        pd.DataFrame({"distribution": distribution_learned}).to_excel(filename + "distribution.xlsx", index=False, header=False)
        
    display_finish()


# UTIL ------------------------------------------------------------------------ #
def display_finish() -> None:
    print("------------------------------------------------------------------------")
    print("FINISHED")
    print("------------------------------------------------------------------------")


if __name__ == "__main__":
    #subprocess.call(["flake8", "."])
    main()
