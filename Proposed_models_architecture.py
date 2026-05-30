n_qubits = len(features)
qdev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(qdev, interface="torch")
def circuit_baseline(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]



#HQC-Seq model
class ProposedHybridQNN(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.prenet = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits)
        )
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit_baseline, weight_shapes)
        self.postnet = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.prenet(x)
        x = self.qlayer(x) * 5.0
        return self.postnet(x)

@qml.qnode(qdev, interface="torch")
def circuit_strong(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]



#HQC-Par model
class ProposedHybridQNN_v2(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.classical_path = nn.Sequential(
            nn.Linear(len(features), 8),
            nn.ReLU(),
            nn.Linear(8, n_qubits)
        )
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_path = qml.qnn.TorchLayer(circuit_strong, weight_shapes)
        self.postnet = nn.Sequential(
            nn.Linear(n_qubits * 2, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out_c = self.classical_path(x)
        out_q = self.quantum_path(x) * 5.0
        return self.postnet(torch.cat((out_c, out_q), dim=1))
