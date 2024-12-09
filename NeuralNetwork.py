import math
import random
import csv
import tkinter as tk
from tkinter import ttk
import sys
from statistics import median

# Global configuration (updated at runtime)
activation_choice_hidden = "sigmoid"  # Activation for hidden layers
activation_choice_output = "sigmoid"  # Activation for output layer
cost_function_choice = "crossentropy"  # Options: "mse", "crossentropy"
learning_rate = 0.1

hidden_layer_structure = [8, 8]


###################################
# Activation Functions and Derivatives
###################################
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


def activation_forward(x, activation_type):
    if activation_type == "sigmoid":
        return sigmoid(x)
    elif activation_type == "tanh":
        return tanh_act(x)
    elif activation_type == "relu":
        return relu(x)
    else:
        return sigmoid(x)


def activation_derivative(output, raw_x, activation_type):
    if activation_type == "sigmoid":
        return d_sigmoid(output)
    elif activation_type == "tanh":
        return d_tanh(output)
    elif activation_type == "relu":
        return d_relu(raw_x)
    else:
        return d_sigmoid(output)


# Cost Functions
def cost_derivative(output, target):
    if cost_function_choice == "mse":
        return output - target
    elif cost_function_choice == "crossentropy":
        epsilon = 1e-9
        return (output - target) / ((output + epsilon) * (1 - output + epsilon))
    else:
        return output - target


class Axon:
    def __init__(self, in_n, out_n, weight=None):
        self.input = in_n
        self.output = out_n
        self.weight = random.uniform(-0.1, 0.1) if weight is None else weight


class Neuron:
    def __init__(self, x, y, input_idx=-1, bias=None, activation_type="sigmoid"):
        self.x = x
        self.y = y
        self.index = input_idx
        self.bias = random.uniform(-0.1, 0.1) if bias is None else bias
        self.result = None
        self.error = 0.0
        self.inputs = []
        self.outputs = []
        self.raw_input_val = 0.0
        self.activation_type = activation_type

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
            self.result = inputs[self.index]
            self.raw_input_val = self.result
        else:
            total = self.bias
            for in_axon in self.inputs:
                in_n = in_axon.input
                in_val = in_n.forward_prop(inputs) * in_axon.weight
                total += in_val
            self.raw_input_val = total
            self.result = activation_forward(total, self.activation_type)
        return self.result

    def back_prop(self):
        grad_activation = activation_derivative(
            self.result, self.raw_input_val, self.activation_type
        )
        delta = self.error * grad_activation
        global learning_rate
        self.bias -= learning_rate * delta
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
    def __init__(
        self,
        num_inputs,
        hidden_structure,
        num_outputs,
        activation_hidden,
        activation_output,
    ):
        self.inputs = []
        self.hidden_layers = []
        self.outputs = []
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output

        # Create input layer
        for idx in range(num_inputs):
            in_n = Neuron(0, 0, idx, activation_type=self.activation_hidden)
            self.inputs.append(in_n)

        prev_layer = self.inputs

        # Create hidden layers
        for layer_size in hidden_structure:
            current_layer = []
            for _ in range(layer_size):
                neuron = Neuron(0, 0, activation_type=self.activation_hidden)
                for prev_neuron in prev_layer:
                    neuron.connect_input(prev_neuron)
                    prev_neuron.connect_output(neuron)
                current_layer.append(neuron)
            self.hidden_layers.append(current_layer)
            prev_layer = current_layer

        # Create output layer
        for _ in range(num_outputs):
            out_n = Neuron(0, 0, activation_type=self.activation_output)
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
        for idx, out_n in enumerate(self.outputs):
            out_n.error = cost_derivative(out_n.result, target_outputs[idx])
        for out_n in self.outputs:
            out_n.back_prop()

    def train(self, data_point):
        self.forward_prop(data_point.inputs)
        self.back_prop(data_point.outputs)

    def test(self, data_points):
        correct = 0
        total = len(data_points)
        for dp in data_points:
            out = self.predict(dp.inputs)
            pred = 1 if out > 0.5 else 0
            if pred == dp.outputs[0]:
                correct += 1
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def predict(self, inputs):
        self.forward_prop(inputs)
        return self.outputs[0].result

    def log_weight_bias_summary(self):
        all_neurons = self.get_all_neurons()
        total_weights = 0
        weight_count = 0
        total_bias = 0
        for n in all_neurons:
            total_bias += n.bias
            for ax in n.inputs:
                total_weights += ax.weight
                weight_count += 1
        avg_bias = total_bias / len(all_neurons)
        avg_weight = total_weights / weight_count if weight_count > 0 else 0.0
        print(
            f"Weight/Bias Summary: Avg Weight={avg_weight:.4f}, Avg Bias={avg_bias:.4f}"
        )

    def log_some_weights(self):
        # Print some weights from each layer to show they are changing:
        print("Logging sample weights from each layer:")

        all_layers = [self.inputs] + self.hidden_layers + [self.outputs]
        for li in range(len(all_layers) - 1):
            layer_name = (
                "Input->Hidden"
                if li == 0
                else (
                    "Hidden->Hidden" if li < len(all_layers) - 2 else "Hidden->Output"
                )
            )
            current_layer = all_layers[li]
            next_layer = all_layers[li + 1]
            if len(next_layer) > 0 and len(next_layer[0].inputs) > 0:
                # Just print the first few axons of the first neuron in the next layer
                print(f" {layer_name} Layer:")
                target_neuron = next_layer[0]
                for i, ax in enumerate(target_neuron.inputs[:3]):
                    print(f"  Weight {i}: {ax.weight:.4f}")
            else:
                print(f" {layer_name} Layer: No neurons to log.")


class DataPoint:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


def load_titanic_data(filename="train.csv", train_ratio=0.8):
    rows = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if len(rows) == 0:
        raise ValueError("No data found in the file.")

    if "Age" not in rows[0]:
        raise ValueError(
            "The dataset does not contain an 'Age' column. Available columns: {}".format(
                rows[0].keys()
            )
        )

    # Extract features: Pclass, Sex, Age, SibSp, Parch, Fare
    ages = []
    fares = []
    for r in rows:
        if r["Age"] != "":
            ages.append(float(r["Age"]))
        if r["Fare"] != "":
            fares.append(float(r["Fare"]))
    median_age = median(ages) if ages else 30.0
    median_fare = median(fares) if fares else 14.4

    data_points = []
    for r in rows:
        try:
            pclass = float(r["Pclass"])
            sex = 1.0 if r["Sex"] == "male" else 0.0
            age = float(r["Age"]) if r["Age"] != "" else median_age
            sibsp = float(r["SibSp"])
            parch = float(r["Parch"])
            fare = float(r["Fare"]) if r["Fare"] != "" else median_fare

            # Normalize
            age /= 80.0
            fare /= 512.0
            survived = float(r["Survived"])

            inputs = [pclass, sex, age, sibsp, parch, fare]
            outputs = [survived]
            data_points.append(DataPoint(inputs, outputs))
        except:
            continue

    random.shuffle(data_points)
    cutoff = int(len(data_points) * train_ratio)
    train_data = data_points[:cutoff]
    test_data = data_points[cutoff:]
    return train_data, test_data


##########################################################
# GUI with modifications to show node outputs and ensure all layers update
##########################################################


class NetworkGUI:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        self.root = tk.Tk()
        self.root.title("Neural Network Visualization")

        # Main frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Layer structure
        tk.Label(control_frame, text="Hidden Layers (comma-separated sizes):").pack(
            side=tk.LEFT
        )
        self.layer_entry = tk.Entry(control_frame, width=10)
        self.layer_entry.insert(0, "8,8")
        self.layer_entry.pack(side=tk.LEFT)

        # Activation for hidden layers
        tk.Label(control_frame, text="Hidden Activation:").pack(side=tk.LEFT)
        self.hidden_act_var = tk.StringVar(value="sigmoid")
        hidden_act_box = ttk.Combobox(
            control_frame,
            textvariable=self.hidden_act_var,
            values=["sigmoid", "tanh", "relu"],
            width=8,
        )
        hidden_act_box.pack(side=tk.LEFT)

        # Activation for output layer
        tk.Label(control_frame, text="Output Activation:").pack(side=tk.LEFT)
        self.output_act_var = tk.StringVar(value="sigmoid")
        output_act_box = ttk.Combobox(
            control_frame,
            textvariable=self.output_act_var,
            values=["sigmoid", "tanh", "relu"],
            width=8,
        )
        output_act_box.pack(side=tk.LEFT)

        # Cost Function
        tk.Label(control_frame, text="Cost:").pack(side=tk.LEFT)
        self.cost_var = tk.StringVar(value="crossentropy")
        cost_box = ttk.Combobox(
            control_frame,
            textvariable=self.cost_var,
            values=["mse", "crossentropy"],
            width=12,
        )
        cost_box.pack(side=tk.LEFT)

        # Epochs
        tk.Label(control_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_entry = tk.Entry(control_frame, width=5)
        self.epochs_entry.insert(0, "5")
        self.epochs_entry.pack(side=tk.LEFT)

        # Learning Rate
        tk.Label(control_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_entry = tk.Entry(control_frame, width=5)
        self.lr_entry.insert(0, "0.1")
        self.lr_entry.pack(side=tk.LEFT)

        # Buttons
        tk.Button(control_frame, text="Build Network", command=self.build_network).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(control_frame, text="Train Network", command=self.train_network).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(control_frame, text="Test Network", command=self.test_network).pack(
            side=tk.LEFT, padx=5
        )

        self.canvas = tk.Canvas(self.root, width=1000, height=600, bg="white")
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.network = None
        self.neuron_positions = []

    def parse_layer_structure(self):
        txt = self.layer_entry.get().strip()
        if not txt:
            return []
        return [int(x.strip()) for x in txt.split(",") if x.strip().isdigit()]

    def build_network(self):
        global cost_function_choice, activation_choice_hidden, activation_choice_output, hidden_layer_structure, learning_rate
        hidden_layer_structure = self.parse_layer_structure()
        activation_choice_hidden = self.hidden_act_var.get()
        activation_choice_output = self.output_act_var.get()
        cost_function_choice = self.cost_var.get()

        try:
            learning_rate = float(self.lr_entry.get().strip())
        except:
            learning_rate = 0.1

        # Create network
        self.network = Network(
            6,
            hidden_layer_structure,
            1,
            activation_choice_hidden,
            activation_choice_output,
        )

        # Lay out the network
        self._layout_network()
        self.draw_network()

        print("Network built with structure:", hidden_layer_structure)
        print(
            "Hidden activation:",
            activation_choice_hidden,
            "Output activation:",
            activation_choice_output,
            "Cost:",
            cost_function_choice,
        )
        print(f"Learning Rate: {learning_rate}")

    def _layout_network(self):
        self.neuron_positions = []
        if not self.network:
            return

        x_offset = 100
        y_spacing = 50

        layers = (
            [self.network.inputs] + self.network.hidden_layers + [self.network.outputs]
        )
        layer_x = x_offset
        for layer in layers:
            layer_height = len(layer) * y_spacing
            y_start = 300 - layer_height / 2
            layer_positions = []
            for i, neuron in enumerate(layer):
                x = layer_x
                y = y_start + i * y_spacing
                layer_positions.append((x, y))
            self.neuron_positions.append(layer_positions)
            layer_x += 200

    def draw_network(self):
        self.canvas.delete("all")
        if not self.network:
            return

        all_layers = (
            [self.network.inputs] + self.network.hidden_layers + [self.network.outputs]
        )
        # Draw neurons
        for li, layer_positions in enumerate(self.neuron_positions):
            for x, y in layer_positions:
                self.canvas.create_oval(
                    x - 15, y - 15, x + 15, y + 15, fill="lightblue", outline="black"
                )

        # Draw axons with weights
        # Reduced scaling factor for line thickness to highlight small changes
        for li in range(len(self.neuron_positions) - 1):
            for ni, (x1, y1) in enumerate(self.neuron_positions[li]):
                in_neuron = all_layers[li][ni]
                for oi, (x2, y2) in enumerate(self.neuron_positions[li + 1]):
                    out_neuron = all_layers[li + 1][oi]
                    for axon in out_neuron.inputs:
                        if axon.input is in_neuron:
                            weight = axon.weight
                            color = "red" if weight > 0 else "blue"
                            # smaller multiplier so even small changes are visible
                            width = min(max(abs(weight) * 10, 1), 5)
                            self.canvas.create_line(
                                x1 + 15, y1, x2 - 15, y2, fill=color, width=width
                            )
                            mid_x = (x1 + 15 + x2 - 15) / 2
                            mid_y = (y1 + y2) / 2
                            self.canvas.create_text(
                                mid_x,
                                mid_y,
                                text=f"{weight:.2f}",
                                fill="black",
                                font=("Arial", 8),
                            )

        # Draw biases
        for li, layer_positions in enumerate(self.neuron_positions):
            layer = all_layers[li]
            for ni, (x, y) in enumerate(layer_positions):
                neuron = layer[ni]
                self.canvas.create_text(
                    x,
                    y - 25,
                    text=f"b={neuron.bias:.2f}",
                    fill="black",
                    font=("Arial", 8),
                )

        self.root.update()

    def display_neuron_outputs(self, sample_inputs):
        # After training or testing, we can forward_prop a sample input and show neuron outputs
        self.network.forward_prop(sample_inputs)
        all_layers = (
            [self.network.inputs] + self.network.hidden_layers + [self.network.outputs]
        )

        # Draw neuron outputs next to each neuron
        for li, layer_positions in enumerate(self.neuron_positions):
            layer = all_layers[li]
            for ni, (x, y) in enumerate(layer_positions):
                neuron = layer[ni]
                # neuron.result should hold the activation after forward_prop
                self.canvas.create_text(
                    x + 30,
                    y,
                    text=f"{neuron.result:.2f}",
                    fill="green",
                    font=("Arial", 8),
                )
        self.root.update()

    def train_network(self):
        if not self.network:
            print("Build the network first!")
            return
        try:
            epochs = int(self.epochs_entry.get().strip())
        except:
            epochs = 5

        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            random.shuffle(self.train_data)
            for data_point in self.train_data:
                self.network.train(data_point)

            # Evaluate accuracy on test set
            accuracy = self.network.test(self.test_data)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy*100:.2f}%")

            # Log sample predictions from training data
            print("Sample predictions on training data:")
            for i in range(min(3, len(self.train_data))):
                sample_output = self.network.predict(self.train_data[i].inputs)
                print(
                    f" Sample {i} input={self.train_data[i].inputs} pred={sample_output:.4f} target={self.train_data[i].outputs[0]}"
                )

            # Log weight and bias summary + multiple layers sample weights
            self.network.log_weight_bias_summary()
            self.network.log_some_weights()

            # Update GUI
            self.draw_network()

            # Also display neuron outputs from a sample input to see internal values changing:
            if len(self.train_data) > 0:
                sample_input = self.train_data[0].inputs
                self.display_neuron_outputs(sample_input)

        print("Training completed.")

    def test_network(self):
        if not self.network:
            print("Build the network first!")
            return
        # Test accuracy on test_data
        accuracy = self.network.test(self.test_data)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        print("Sample predictions on test data:")
        for i in range(min(3, len(self.test_data))):
            output = self.network.predict(self.test_data[i].inputs)
            print(
                f" Test sample {i} input={self.test_data[i].inputs}, pred={output:.4f}, target={self.test_data[i].outputs[0]}"
            )

        # Show neuron outputs after testing:
        if len(self.test_data) > 0:
            sample_input = self.test_data[0].inputs
            self.display_neuron_outputs(sample_input)

    def mainloop(self):
        self.root.mainloop()


def main():
    train_data, test_data = load_titanic_data()
    gui = NetworkGUI(train_data, test_data)
    gui.mainloop()


if __name__ == "__main__":
    main()
