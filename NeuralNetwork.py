import math
import random
import csv
import tkinter as tk
from tkinter import ttk
import sys
from statistics import median

# Global configuration (can be changed at runtime)
activation_choice = "sigmoid"  # Options: "sigmoid", "tanh", "relu"
cost_function_choice = "crossentropy"  # Options: "mse", "crossentropy"
learning_rate = 0.1


# Activation Functions and Their Derivatives
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def d_sigmoid(output):
    return output * (1.0 - output)


def tanh_act(x):
    return math.tanh(x)


def d_tanh(output):
    return 1.0 - (output**2)


def relu(x):
    return x if x > 0 else 0


def d_relu(raw_x):
    return 1 if raw_x > 0 else 0


def activation_forward(x):
    if activation_choice == "sigmoid":
        return sigmoid(x)
    elif activation_choice == "tanh":
        return tanh_act(x)
    elif activation_choice == "relu":
        return relu(x)
    else:
        return sigmoid(x)


def activation_derivative(output, raw_x):
    # For sigmoid and tanh, derivative uses the output value directly
    if activation_choice == "sigmoid":
        return d_sigmoid(output)
    elif activation_choice == "tanh":
        return d_tanh(output)
    elif activation_choice == "relu":
        return d_relu(raw_x)
    else:
        return d_sigmoid(output)


# Cost Functions
def cost_derivative(output, target):
    if cost_function_choice == "mse":
        # d/dOut of MSE = (output - target)
        return output - target
    elif cost_function_choice == "crossentropy":
        # Cross entropy derivative with sigmoid:
        epsilon = 1e-9
        return (output - target) / ((output + epsilon) * (1 - output + epsilon))
    else:
        # Default to MSE if unknown
        return output - target


class Axon:
    def __init__(self, in_n, out_n, weight=None):
        self.input = in_n
        self.output = out_n
        self.weight = random.uniform(-0.1, 0.1) if weight is None else weight


class Neuron:
    def __init__(self, x, y, input_idx=-1, bias=None):
        self.x = x
        self.y = y
        self.index = input_idx
        self.bias = random.uniform(-0.1, 0.1) if bias is None else bias
        self.result = None
        self.error = 0.0
        self.inputs = []
        self.outputs = []
        self.raw_input_val = 0.0

    def connect_input(self, in_n):
        in_axon = Axon(in_n, self)
        self.inputs.append(in_axon)

    def connect_output(self, out_n):
        out_axon = Axon(self, out_n)
        self.outputs.append(out_axon)

    def forward_prop(self, inputs):
        if self.result is not None:
            return self.result
        if self.index >= 0:
            # Input neuron
            self.result = inputs[self.index]
            self.raw_input_val = self.result
        else:
            total = self.bias
            for in_axon in self.inputs:
                in_n = in_axon.input
                in_val = in_n.forward_prop(inputs) * in_axon.weight
                total += in_val
            self.raw_input_val = total
            self.result = activation_forward(total)
        return self.result

    def back_prop(self):
        grad_activation = activation_derivative(self.result, self.raw_input_val)
        delta = self.error * grad_activation
        # Update bias
        self.bias -= learning_rate * delta
        # Update weights and propagate error backward
        for in_axon in self.inputs:
            in_n = in_axon.input
            in_n.error += delta * in_axon.weight
            in_axon.weight -= learning_rate * delta * in_n.result
        self.error = 0.0

    def reset(self):
        self.result = None
        self.error = 0.0
        self.raw_input_val = 0.0


class Network:
    def __init__(self, num_inputs, num_hidden_layers, hidden_layer_width, num_outputs):
        self.inputs = []
        self.hidden_layers = []
        self.outputs = []

        # Create input layer
        for idx in range(num_inputs):
            in_n = Neuron(0, 0, idx)
            self.inputs.append(in_n)

        prev_layer = self.inputs

        # Create hidden layers
        for _ in range(num_hidden_layers):
            current_layer = []
            for _ in range(hidden_layer_width):
                neuron = Neuron(0, 0)
                for prev_neuron in prev_layer:
                    neuron.connect_input(prev_neuron)
                    prev_neuron.connect_output(neuron)
                current_layer.append(neuron)
            self.hidden_layers.append(current_layer)
            prev_layer = current_layer

        # Create output layer
        for _ in range(num_outputs):
            out_n = Neuron(0, 0)
            for prev_neuron in prev_layer:
                out_n.connect_input(prev_neuron)
                prev_neuron.connect_output(out_n)
            self.outputs.append(out_n)

    def get_all_neurons(self):
        neurons = self.inputs[:]
        for layer in self.hidden_layers:
            neurons.extend(layer)
        neurons.extend(self.outputs)
        return neurons

    def forward_prop(self, inputs):
        for neuron in self.get_all_neurons():
            neuron.reset()
        for out_n in self.outputs:
            out_n.forward_prop(inputs)

    def back_prop(self, target_outputs):
        # Compute errors at output
        for idx, out_n in enumerate(self.outputs):
            out_n.error = cost_derivative(out_n.result, target_outputs[idx])
        # Backprop
        for out_n in self.outputs:
            out_n.back_prop()

    def train(self, data_point):
        self.forward_prop(data_point.inputs)
        self.back_prop(data_point.outputs)

    def test(self, data_point):
        self.forward_prop(data_point.inputs)
        return [out_n.result for out_n in self.outputs]


class DataPoint:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


##########################################################
# Step 1: Load and Preprocess Titanic Dataset
##########################################################


def load_titanic_data(filename="train.csv", train_ratio=0.8):
    # We'll use features: Pclass, Sex, Age, SibSp, Parch, Fare
    # Target: Survived (0 or 1)
    rows = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Debug print
        print("CSV columns:", reader.fieldnames)

        for row in reader:
            rows.append(row)

    # Check if 'Age' is in the columns:
    if "Age" not in rows[0]:
        raise ValueError(
            "The dataset does not contain an 'Age' column. Available columns: {}".format(
                rows[0].keys()
            )
        )

    # Extract relevant features and handle missing values
    # Convert Sex to binary: male=1, female=0
    # Fill missing Age and Fare with median
    ages = []
    fares = []
    for r in rows:
        if r["Age"] != "":
            ages.append(float(r["Age"]))
        if r["Fare"] != "":
            fares.append(float(r["Fare"]))
    median_age = median(ages)
    median_fare = median(fares)

    data_points = []
    for r in rows:
        try:
            pclass = float(r["Pclass"])
            sex = 1.0 if r["Sex"] == "male" else 0.0
            age = float(r["Age"]) if r["Age"] != "" else median_age
            sibsp = float(r["SibSp"])
            parch = float(r["Parch"])
            fare = float(r["Fare"]) if r["Fare"] != "" else median_fare
            survived = float(r["Survived"])

            # Normalize some continuous inputs (for simplicity, we can divide Age, Fare by max)
            # This is not strictly required, but can help training.
            # Just a simple scaling:
            age /= 80.0  # 80 is roughly max age
            fare /= 512.0  # 512 is roughly max fare in Titanic dataset

            inputs = [pclass, sex, age, sibsp, parch, fare]
            outputs = [survived]
            data_points.append(DataPoint(inputs, outputs))
        except:
            # If any parsing fails, skip that row
            continue

    # Shuffle and split
    random.shuffle(data_points)
    cutoff = int(len(data_points) * train_ratio)
    train_data = data_points[:cutoff]
    test_data = data_points[cutoff:]

    return train_data, test_data


##########################################################
# Step 5: GUI to Display the Network
##########################################################


class NetworkGUI:
    def __init__(self, network):
        self.network = network
        self.root = tk.Tk()
        self.root.title("Neural Network Visualization")

        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.pack()

        # Create dropdowns or entries to configure activation/cost if desired
        control_frame = tk.Frame(self.root)
        control_frame.pack()

        tk.Label(control_frame, text="Activation:").pack(side=tk.LEFT)
        self.activation_var = tk.StringVar(value=activation_choice)
        activation_box = ttk.Combobox(
            control_frame,
            textvariable=self.activation_var,
            values=["sigmoid", "tanh", "relu"],
        )
        activation_box.pack(side=tk.LEFT)

        tk.Label(control_frame, text="Cost:").pack(side=tk.LEFT)
        self.cost_var = tk.StringVar(value=cost_function_choice)
        cost_box = ttk.Combobox(
            control_frame, textvariable=self.cost_var, values=["mse", "crossentropy"]
        )
        cost_box.pack(side=tk.LEFT)

        tk.Button(control_frame, text="Update Config", command=self.update_config).pack(
            side=tk.LEFT
        )

        # Layout neurons on canvas:
        self.neuron_positions = []
        self._layout_network()

    def _layout_network(self):
        # We will place input layer neurons, then hidden layers, then output layer
        x_offset = 100
        y_spacing = 50

        layers = (
            [self.network.inputs] + self.network.hidden_layers + [self.network.outputs]
        )

        max_layer_size = max(len(layer) for layer in layers)
        start_y = 300 - (max_layer_size * y_spacing) / 2

        # Compute positions
        layer_x = x_offset
        for layer in layers:
            layer_positions = []
            layer_height = len(layer) * y_spacing
            y_start = 300 - layer_height / 2
            for i, neuron in enumerate(layer):
                x = layer_x
                y = y_start + i * y_spacing
                layer_positions.append((x, y))
            self.neuron_positions.append(layer_positions)
            layer_x += 200

    def draw_network(self):
        self.canvas.delete("all")
        # Draw neurons
        for li, layer_positions in enumerate(self.neuron_positions):
            for ni, (x, y) in enumerate(layer_positions):
                self.canvas.create_oval(
                    x - 15, y - 15, x + 15, y + 15, fill="lightblue", outline="black"
                )

        # Draw axons
        # For each layer except the last, connect to next layer
        for li in range(len(self.neuron_positions) - 1):
            for ni, (x1, y1) in enumerate(self.neuron_positions[li]):
                neuron = (
                    self.network.inputs
                    if li == 0
                    else (
                        self.network.hidden_layers[li - 1]
                        if li <= len(self.network.hidden_layers)
                        else self.network.outputs
                    )
                )[ni]
                if li == 0:
                    current_layer = self.network.inputs
                elif li == len(self.network.hidden_layers):
                    current_layer = self.network.hidden_layers[-1]
                else:
                    current_layer = self.network.hidden_layers[li - 1]

                # Actually we need to get layer neurons:
                if li == 0:
                    current_layer = self.network.inputs
                elif li <= len(self.network.hidden_layers):
                    current_layer = self.network.hidden_layers[li - 1]
                else:
                    current_layer = self.network.outputs

                # next layer
                if li < len(self.network.hidden_layers):
                    next_layer = self.network.hidden_layers[li]
                else:
                    next_layer = self.network.outputs

                in_neuron = current_layer[ni]
                for oi, (x2, y2) in enumerate(self.neuron_positions[li + 1]):
                    out_neuron = (
                        self.network.hidden_layers[li]
                        if li < len(self.network.hidden_layers)
                        else self.network.outputs
                    )[oi]
                    # find axon weight
                    # an axon from in_neuron to out_neuron
                    for axon in out_neuron.inputs:
                        if axon.input is in_neuron:
                            weight = axon.weight
                            color = "red" if weight > 0 else "blue"
                            width = min(max(abs(weight) * 50, 1), 5)
                            self.canvas.create_line(
                                x1 + 15, y1, x2 - 15, y2, fill=color, width=width
                            )

        # Draw biases as text above each neuron
        all_layers = (
            [self.network.inputs] + self.network.hidden_layers + [self.network.outputs]
        )
        for li, layer_positions in enumerate(self.neuron_positions):
            for ni, (x, y) in enumerate(layer_positions):
                neuron = all_layers[li][ni]
                self.canvas.create_text(
                    x,
                    y - 25,
                    text=f"b={neuron.bias:.2f}",
                    fill="black",
                    font=("Arial", 8),
                )

        self.root.update()

    def update_config(self):
        global activation_choice, cost_function_choice
        activation_choice = self.activation_var.get()
        cost_function_choice = self.cost_var.get()
        print(
            f"Updated config: Activation={activation_choice}, Cost={cost_function_choice}"
        )

    def mainloop(self):
        self.root.mainloop()


##########################################################
# Step 6: Run Training and Observe Updates
##########################################################


def main():
    # Load data
    train_data, test_data = load_titanic_data()

    # Create network
    # Let's say we have 6 inputs (Pclass, Sex, Age, SibSp, Parch, Fare),
    # 2 hidden layers with width 8, and 1 output (Survived probability).
    num_inputs = 6
    num_hidden_layers = 2
    hidden_layer_width = 8
    num_outputs = 1

    network = Network(num_inputs, num_hidden_layers, hidden_layer_width, num_outputs)

    # Create GUI
    gui = NetworkGUI(network)
    gui.draw_network()

    # Train for some epochs
    epochs = 5
    for epoch in range(epochs):
        # Shuffle training data
        random.shuffle(train_data)
        for data_point in train_data:
            network.train(data_point)
        # After each epoch, test accuracy
        correct = 0
        total = len(test_data)
        for data_point in test_data:
            output = network.test(data_point)[0]
            prediction = 1 if output > 0.5 else 0
            if prediction == data_point.outputs[0]:
                correct += 1
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy*100:.2f}%")
        # Update visualization
        gui.draw_network()

    gui.mainloop()


if __name__ == "__main__":
    main()
