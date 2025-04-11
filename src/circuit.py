import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Tuple
import pennylane as qml
from enum import Enum

class CircuitType(Enum):
    ''' What circuit to compile to '''
    PENNYLANE = 0
    QISKIT = 1

class Gate(ABC):
    is_ctr = False
    dim = 1
    is_placeholder = False
    @property
    def m(self) -> np.ndarray:
        '''the unitary matrix representation of the gate'''

class CNOT(Gate):
    def __init__(self, ctr:int, target:int) -> None:
        super().__init__()
        self.ctr = ctr
        self.target = target
        self.is_ctr = True
        self.dim = 4
    @property
    def m(self) -> np.ndarray:
        return np.array([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,0,1],
                [0,0,1,0]
        ])
    def pennylane(self):
        return qml.CNOT([self.ctr, self.target])

class I(Gate):
    is_placeholder = True
    def __new__(cls):
        return super().__new__()
    @property
    def m(cls) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, 1]])
    @classmethod
    def pennylane(cls, wire:int):
        return qml.Identity(wire)

class H(Gate):
    def __new__(cls):
        return super().__new__()
    @property
    def m(cls) -> np.ndarray:
        return 1/np.sqrt(2) * np.array([
            [1, 1],
            [1, -1]])
    @classmethod
    def pennylane(cls, wire:int):
        return qml.Hadamard(wire)

class S(Gate):
    is_placeholder = False
    def __new__(cls):
        return super().__new__()
    @property
    def m(cls) -> np.ndarray:
        return NotImplementedError
    @classmethod
    def pennylane(cls, wire:int):
        return qml.S(wire)

class T(Gate):
    is_placeholder = False
    def __new__(cls):
        return super().__new__()
    @property
    def m(cls) -> np.ndarray:
        return NotImplementedError
    @classmethod
    def pennylane(cls, wire:int):
        return qml.T(wire)

class X(Gate):
    is_placeholder = False
    def __new__(cls):
        return super().__new__()
    @property
    def m(cls) -> np.ndarray:
        return NotImplementedError
    @classmethod
    def pennylane(cls, wire:int):
        return qml.PauliX(wire)

class RX(Gate):
    def __init__(self, angle:float):
        super().__init__()
        self.angle = angle
    @property
    def m(cls) -> np.ndarray:
        raise NotImplementedError
    def pennylane(self, wire:int):
        return qml.RX(self.angle, wire)

class RY(Gate):
    def __init__(self, angle:float):
        super().__init__()
        self.angle = angle
    @property
    def m(cls) -> np.ndarray:
        raise NotImplementedError
    def pennylane(self, wire:int):
        return qml.RY(self.angle, wire)
    
class RZ(Gate):
    def __init__(self, angle:float):
        super().__init__()
        self.angle = angle
    @property
    def m(cls) -> np.ndarray:
        raise NotImplementedError
    def pennylane(self, wire:int):
        return qml.RZ(self.angle, wire)


class CircuitBase(ABC):
    def __init__(self, width:int, circuit_api:CircuitType=CircuitType.PENNYLANE, shots:Union[int,None]=None) -> None:
        self.width = width # number of wires
        self.shots = shots # number of (simulation/evaluation) shots
        self.gate_table = {} #{layer:[I for _ in range(self.width)] for layer in range(self.depth)}
        self.circuit_api = circuit_api
        #self.status()
        self.matrix_table = None

    def insert_matrix_layer(self, encoding:np.ndarray):
        assert len(encoding) == self.width, "Encoding dimension must match circuit width"
        if self.matrix_table is not None:
            self.matrix_table = np.c_[self.matrix_table, encoding]
        else:
            self.matrix_table = encoding

    def get_matrix_circuit(self):
        return self.matrix_table
    
    def set_matrix_circuit(self, new_table:np.ndarray):
        assert new_table.shape[0] == self.width
        self.matrix_table = new_table

    def get_matrix_layer(self, layer):
        return self.matrix_table[::,layer]

    def get_all_matrix_layers(self):
        return self.matrix_table.T

    def __len__(self)->int:
        return len(self.gate_table)

    def status(self):
        if self.circuit_api == CircuitType.PENNYLANE:
            qml.about()

    def reset(self):
        self.gate_table = {}

    def insert_gate_layer(self, layer:int, gates:List[Gate]):
        assert len(gates) == self.width
        self.gate_table[layer] = gates

    def insert_gate(self, layer:int, wire, gate) -> None:
        if not (layer in self.gate_table):
            # check key exists, e.g. 0 in {layer:[gates]}
            self.gate_table[layer] = [I for _ in range(self.width)]
        self.gate_table[layer][wire] = gate
    
    def remove_layer(self, layer) -> None:
        del self.gate_table[layer]

    @abstractmethod
    def make_circuit():
        raise NotImplementedError

    def to_pennylane(self):
        # 2) Instantiate the q-note into a circuit object 
        dev = qml.device("lightning.qubit", wires=self.width, shots=self.shots)
        circuit = qml.QNode(self.make_circuit, dev, interface=None)
        return circuit
    
    def draw(self, *params):
        if self.circuit_api == CircuitType.PENNYLANE:
            return qml.draw(self.to_pennylane())(*params)
    
    def eval(self, *params):
        if self.circuit_api == CircuitType.PENNYLANE:
            return self.to_pennylane()(*params)


class Optimization_Circuit(CircuitBase):
    def __init__(self, width:int, circuit_api:CircuitType=CircuitType.PENNYLANE, shots:Union[int,None]=None, seed:int=0) -> None:
        super().__init__(width, circuit_api, shots)
    
    def reset(self):
        super().reset()

    def state(self):
        return self.to_pennylane()()
    
    def make_circuit(self, inputs:Union[None, List[int]]=None):
        for layer, gate_list in sorted(self.gate_table.items()):
            for wire, gate in enumerate(gate_list):
                if gate.is_placeholder:
                    continue
                if gate.is_ctr:
                    gate.pennylane()
                elif gate.dim > 1:
                    raise ValueError
                else:
                    gate.pennylane(wire)
            qml.Barrier()
        if isinstance(inputs, Tuple) and len(inputs)==2:
            return qml.expval(qml.PauliZ(inputs[0]) @ qml.PauliZ(inputs[1]))
        elif isinstance(inputs, int):
            return qml.expval(qml.PauliZ(inputs))
        elif inputs is None:
            return qml.sample()
        else:
            raise ValueError

    def eval(self, inputs):
        return self.to_pennylane()(inputs)

class MaxCut_Circuit(Optimization_Circuit):
    def make_circuit(self, inputs:Union[None, List[int]]=None):
        for layer, gate_list in self.gate_table.items():
            for wire, gate in enumerate(gate_list):
                if gate.is_placeholder:
                    continue
                if gate.is_ctr:
                    gate.pennylane()
                elif gate.dim > 1:
                    raise ValueError
                else:
                    gate.pennylane(wire)
            qml.Barrier()
        if isinstance(inputs, Tuple) and len(inputs)==2:
            return qml.expval(qml.PauliZ(inputs[0]) @ qml.PauliZ(inputs[1]))
        elif isinstance(inputs, int):
            return qml.expval(qml.PauliZ(inputs))
        elif inputs is None:
            return qml.sample()
        else:
            raise ValueError

    def eval(self):
        return sum([0.5 * (1 - self.to_pennylane()(inputs=edge)) for edge in self.problem])