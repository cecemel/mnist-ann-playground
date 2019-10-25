import os
import numpy as np
import imageio
import math
from functools import reduce
import copy
import datetime
import multiprocessing
from multiprocessing import Pool
import random
#import pdb; pdb.set_trace()


def load_model(base_dir):
    weights = {}
    for file in os.listdir(base_dir):
        weights[os.path.splitext(file)[0]] = np.load(os.path.join(base_dir, file))

    return weights


def save_model(layers):
    base_dir = os.path.join('./output', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(base_dir, exist_ok=True)
    for l in layers:
        file_path = os.path.join(base_dir, '%s' % l)
        np.save(file_path, layers[l])


def load_file_paths(path):
    # assumes label/sample.png
    data = []

    for label_folder in os.listdir(path):
        files = os.listdir(os.path.join(path, label_folder))
        labeled_files = [{"label": label_folder, "src": os.path.join(path, label_folder, p)} for p in files]
        data.extend(labeled_files)

    return data


def convert_png_to_grayscale(path):
    return imageio.imread(path)  # note for now they are grayscaled (I assume)


def preprocess(path, limit_sample=None):
    """
    Preprocessing pipeline, loads data,
    gets grayscaled versions, normalize between [0, 1] and flattens the image matrix
    """
    data = load_file_paths(path)
    random.shuffle(data)
    if limit_sample:
        data = get_sample_per_class(data, limit_sample)
    for d in data:
        matr = convert_png_to_grayscale(d["src"])
        matr = np.matrix.flatten(matr)
        matr = matr.reshape(matr.size, 1)
        matr = matr/255  # normalize else is going to suck with activation_function
        d["data"] = matr

    print("finished preprocessing")
    return data


def get_sample_per_class(data, size):
    """
    Give a sample size and data, we take for every class a sample
    """
    grouped_data = {}
    for d in data:
        if not grouped_data.get(d["label"]):
            grouped_data[d["label"]] = [d]
            continue
        if grouped_data[d["label"]] and not len(grouped_data[d["label"]]) >= size:
            grouped_data[d["label"]].append(d)

    flat_data = []

    for k in grouped_data:
        flat_data.extend(grouped_data[k])

    return flat_data


def encode_labels(unique_labels):
    """
    Makes a hot encoded vector from the labels
    """
    # TODO: there must be a smarter way
    base_array = np.zeros(len(unique_labels))
    base_array = base_array.reshape(base_array.size, 1)
    labels = {}
    for idx, val in enumerate(unique_labels):
        encoded = np.copy(base_array)
        encoded[idx] = 1
        labels[val] = encoded
    return labels


def generated_seeded_weight_matrices(input_array_size, label_array_size):
    """
     We seed the weight and bias matrices with random values ranging [-1, 1]
     Note: The hidden layer is 32 nodes big. I read somewhere this is more then sufficient
    """
    layer_1 = np.random.uniform(-1, 1, (32, input_array_size))
    bias_1 = np.random.uniform(-1, 1, (32, 1))

    layer_2 = np.random.uniform(-1, 1, (label_array_size, 32))
    bias_2 = np.random.uniform(-1, 1, (label_array_size, 1))

    return {"layer_1": layer_1, "bias_1": bias_1, "layer_2": layer_2, "bias_2": bias_2}


def sigmoid(x):
    return 1/(1 + math.exp(-1 * x))


def run_data_through_network(input_array, weights):
    """
     Pipeline to run an image data set through the network
    """
    result_1 = np.add(weights["layer_1"].dot(input_array), weights["bias_1"])
    vfunc = np.vectorize(sigmoid)
    activated_result = vfunc(result_1)
    result_2 = np.add(weights["layer_2"].dot(activated_result), weights["bias_2"])
    return {
        "output": softmax(result_2),
        "result_2": result_2,
        "activated_result": activated_result,
        "result_1": result_1,
        "data": input_array
    }


def calc_gradient_layer_2(result):
    """
     Calculates the gradient for the second layer, given the results from the network for
     one image.
     First calculate delta_z -> note: theta_i is variable of weight matrix.
     E is cross entropy.
     d(E)/d(theta_i) = d(E)/d(u) * d(u)/d(z) * d(z)/d(theta_i)
     where:
        d(E)/d(u) * d(u)/d(z) = p - t = delta_z

    see: https://deepnotes.io/softmax-crossentropy,
         https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
         https://www.youtube.com/watch?v=tIeHLnjs5U8
         Machine Learning with Neural Networks:
           An In-depth Visual Introduction with Python:
           Make Your Own Neural Network in Python:
           A Simple Guide on Machine Learning with Neural Networks (Michael Taylor)
    """
    weights_data = result["weights_data"]
    delta_components = result["output"] - result["truth"]
    delta_components.reshape(delta_components.size)  # make flat array here

    # weights
    gradient_component = np.ones(weights_data["layer_2"].shape)
    for x, y in np.ndindex(weights_data["layer_2"].shape):
        gradient_component[x][y] = delta_components[x] *\
                                   result["activated_result"][y][0]  # because one dimensional vector

    return {"gradient_component": gradient_component,
            "bias_component": np.array(delta_components).reshape(len(delta_components), 1)}


def calc_gradient_layer_1(result):
    """
     Calculates the gradient for the first layer for one result. Very similar as calc_gradient_layer_2
     only more complex.
     The start is the same, then I had to think harder.
     The most useful reference: https://www.youtube.com/watch?v=tIeHLnjs5U8
    """
    weights_data = result["weights_data"]
    delta_components = result["output"] - result["truth"]
    delta_components.reshape(delta_components.size)

    delta_components_1 = []
    for x, y in np.ndindex(result["activated_result"].shape):
        # now calculate d(C)/d(a^l)
        dc_over_d_l_j = np.transpose(weights_data["layer_2"][:, x]).dot(delta_components)
        # make delta from this
        delta_component_1 = dc_over_d_l_j * result["activated_result"][x][0] *\
                            (1 - result["activated_result"][x][0])

        delta_components_1.append(delta_component_1)

    gradient_component = np.ones(weights_data["layer_1"].shape)
    for x, y in np.ndindex(weights_data["layer_1"].shape):
        gradient_component[x, y] = delta_components_1[x] * result["data"][y][0]

    return {"gradient_component": gradient_component,
            "bias_component": np.array(delta_components_1).reshape(len(delta_components_1), 1)}


def update_weights(weights_data, results_data, learning_rate, pool_size=None):
    """
    Procedure to update the weights. Calculates for every result the Gradient matrix.
    The averages these matrices out and updates the weights.
    Since super slow, I put it as a multiprocessing step.
    Probably need some smarter loops.
    """
    pool = multiprocessing.cpu_count() - 1
    if pool_size:
        pool = pool_size

    for r in results_data:
        r["weights_data"] = weights_data

    with Pool(processes=pool) as p:
        gradient_datas = p.map(calc_gradient_layer_2, results_data)

    gradient_components = map(lambda x: x["gradient_component"], gradient_datas)
    bias_components = map(lambda x: x["bias_component"], gradient_datas)
    gradient_layer_2 = reduce(lambda x, y: x + y, gradient_components) / len(results_data)
    bias_layer_2 = reduce(lambda x, y: x + y, bias_components) / len(results_data)
    updated_layer_2 = weights_data["layer_2"] - learning_rate * gradient_layer_2
    updated_bias_2 = weights_data["bias_2"] - learning_rate * bias_layer_2

    with Pool(processes=pool) as p:
        gradient_datas_layer_1 = p.map(calc_gradient_layer_1, results_data)

    gradient_components = map(lambda x: x["gradient_component"], gradient_datas_layer_1)
    bias_components = map(lambda x: x["bias_component"], gradient_datas_layer_1)
    gradient_layer_1 = reduce(lambda x, y: x + y, gradient_components) / len(results_data)
    bias_layer_1 = reduce(lambda x, y: x + y, bias_components) / len(results_data)
    updated_layer_1 = weights_data["layer_1"] - learning_rate * gradient_layer_1
    updated_bias_1 = weights_data["bias_1"] - learning_rate * bias_layer_1

    return {
        "layer_1": updated_layer_1,
        "bias_1": updated_bias_1,
        "layer_2": updated_layer_2,
        "bias_2": updated_bias_2,
        "gradient_layer_1": gradient_layer_1,
        "gradient_layer_2": gradient_layer_2
    }


def check_gradients(old_weights, updated_weights, image_datas, labels_map):
    """
    Procedure to do a numerical check of the gradients. To see wether it makes sense
    If you see a lot of warning, probably somthing is not right.
    There is aso a ratio in the output of the failed ones
    Main resource:
         Machine Learning with Neural Networks:
           An In-depth Visual Introduction with Python:
           Make Your Own Neural Network in Python:
           A Simple Guide on Machine Learning with Neural Networks (Michael Taylor)
    Note: this is only now a hard coded check for gradient_layer_1
    """
    epsilon = 0.0001
    weights_matrix = old_weights["layer_1"]
    gradient_matrix = updated_weights["gradient_layer_1"]
    ok = 0
    nok = 0
    for x, y in np.ndindex(weights_matrix.shape):
        w = copy.deepcopy(old_weights)
        e_plus = np.copy(weights_matrix)
        e_plus[x][y] = e_plus[x][y] + epsilon
        e_min = np.copy(weights_matrix)
        e_min[x][y] = e_min[x][y] - epsilon
        w["layer_1"] = e_plus
        r_plus = loss_samples(run_all_samples_trough_network(image_datas, w, labels_map))
        w["layer_1"] = e_min
        r_min = loss_samples(run_all_samples_trough_network(image_datas, w, labels_map))
        numerical_gradient = (r_plus - r_min)/(2 * epsilon)
        relative_error = (gradient_matrix[x][y] -
                          numerical_gradient) / max(gradient_matrix[x][y], numerical_gradient)
        print(relative_error)
        if(abs(relative_error) > 1e-5):
            nok += 1
            print('----------warning: difference between numerical and analytical gradient exceeds treshold')
        else:
            ok += 1
    print('Results from gradient checking (nok/(ok + nok )):')
    print(nok / (ok + nok))


def run_all_samples_trough_network(image_datas, weights, labels_map):
    """
     All data sets are run through the network.
    """
    results = []
    for d in image_datas:
        result = run_data_through_network(d["data"], weights)
        result["truth"] = labels_map[d["label"]]
        results.append(result)
    return results


def loss_samples(results_data):
    """
    calculates loss. Now is cross entropy.
    """
    error_array = []
    for result in results_data:
        e = - np.sum(np.transpose(result["truth"]).dot(np.log(result["output"])))
        error_array.append(e)
    return mean_error(error_array)


def mean_error(error_array):
    return np.sum(error_array) / len(error_array)


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def count_stats(results):
    total = len(results)
    success = 0
    for r in results:
        if (abs(max(r["output"])[0] - np.transpose(r["truth"]).dot(r["output"])[0])[0]) < 0.000001:
            success += 1

    print('-------------------------Rate (success/total): {}'.format(success/total))


def train(path, limit_sample=None, max_iterations=30, perform_gradient_check=False, error_cut=0.9, weight_update=0.9,
          validation_path=None, pool_size=None):
    """
      Trains data
    """
    image_datas = preprocess(path, limit_sample)
    labels = set([d["label"] for d in image_datas])
    labels_map = encode_labels(labels)
    weights = generated_seeded_weight_matrices(image_datas[0]["data"].size, next(iter(labels_map.values())).size)

    current_weights = weights
    error = 1
    results = None
    iteration = 0
    has_checked_gradients = False
    while error > error_cut and iteration < max_iterations:
        iteration += 1
        results = run_all_samples_trough_network(image_datas, current_weights, labels_map)
        error = loss_samples(results)
        print(error)
        count_stats(results)
        updated_weights = update_weights(current_weights, results, weight_update, pool_size=pool_size)
        current_weights = updated_weights
        if perform_gradient_check and not has_checked_gradients:
            check_gradients(weights, updated_weights, image_datas, labels_map)
            has_checked_gradients = True

    save_model(current_weights)

    if validation_path:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx validation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        image_datas = preprocess(validation_path)
        labels = set([d["label"] for d in image_datas])
        labels_map = encode_labels(labels)
        weights = current_weights
        results = run_all_samples_trough_network(image_datas, weights, labels_map)
        error = loss_samples(results)
        print(error)
        count_stats(results)


def validate(path, weights_path, limit_sample=None):
    """
    Loads the saved model and loads a validation data sets and does basic
    statistic.
    TODO: I have a big problem with numerical stability. Two runs of the same code yield different results.
    """
    image_datas = preprocess(path, limit_sample)
    labels = set([d["label"] for d in image_datas])
    labels_map = encode_labels(labels)
    weights = load_model(weights_path)

    results = run_all_samples_trough_network(image_datas, weights, labels_map)
    error = loss_samples(results)
    print(error)
    count_stats(results)


if __name__ == "__main__":
    train("./input/training", weight_update=1.2, validation_path="./input/testing")
