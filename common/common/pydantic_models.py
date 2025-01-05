
from pydantic import BaseModel
from typing import Optional, Callable, Any
import hashlib
import json

class BaseFEMParameters(BaseModel):


    def to_hash(self) -> str:
        """
        Generates a SHA-256 hash string of the instance's data,
        ensuring all values are JSON serializable.
        """
        # Convert the model to a dictionary
        data = self.model_dump()

        # Ensure all values are JSON serializable
        def make_serializable(value: Any) -> Any:
            if callable(value):
                return f"<callable: {value.__name__}>"
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                return value
            else:
                return str(value)

        # Recursively process the dictionary
        def process_data(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: process_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [process_data(item) for item in obj]
            else:
                return make_serializable(obj)

        # Process the data for serialization
        serializable_data = process_data(data)

        # Serialize to JSON with sorted keys for consistency
        json_data = json.dumps(serializable_data, sort_keys=True)

        # Compute and return the SHA-256 hash
        return hashlib.sha256(json_data.encode('utf-8')).hexdigest()

class ShapeOutlineParameters(BaseFEMParameters):
    x_points: list[list[float]]
    y_points: list[list[float]]
    z_points: list[list[float]]
    arc_center: list[list[list[float]]]
    edge_types: list[list[str]]
    labels: list[list[str]]
    line_loop_groups: list[list[int]]


class BoundaryConstraintConfig(BaseFEMParameters):
    dirichlet_boundary_labels: list[int]
    robin_boundary_labels: list[int]
    uD: Optional[Callable] = None
    sigma: Optional[Callable]
    mu: Optional[Callable] = None

class PDECoefficients(BaseFEMParameters):
    c: Callable
    a: Callable
    f: Callable
    b1: Callable
    b2: Callable
    boundary_constraints: BoundaryConstraintConfig


class TrainingDataPoint(BaseFEMParameters):
    h: float
    shape_parameters: ShapeOutlineParameters
    p: list[list[float]]
    t: list[list[float]]
    e: list[list[float]]
    estimated_u: list[float]

