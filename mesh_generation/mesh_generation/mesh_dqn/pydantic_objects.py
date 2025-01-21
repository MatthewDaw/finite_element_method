from typing import Optional, List, Union, Any
from pydantic import BaseModel
import torch
from torch_geometric.data import Data
from shapely.geometry import Point, Polygon
from common.pydantic_models import ShapeOutlineParameters, ShapeTransformationParameters


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float
    batch_size: int


class EpsilonConfig(BaseModel):
    start: float
    end: float
    decay: int
    gamma: float


class AgentParamsConfig(BaseModel):
    plot_dir: str
    prefix: str
    solver_steps: Optional[int] = None
    save_steps: Optional[int] = None
    NUM_INPUTS: Optional[int] = None
    target_update: int
    num_workers: int
    num_parallel: int
    N_closest: Optional[int] = None
    do_nothing: Optional[bool] = None
    episodes: Optional[int] = None
    goal_vertices: Optional[float] = None
    gt_drag: Optional[float] = None
    gt_time: Optional[float] = None
    p: Optional[float] = None
    smoothing: Optional[bool] = None
    threshold: Optional[float] = None
    expected_avg_improvement: Optional[float] = None
    timesteps: Optional[int] = None
    u: Optional[float] = None
    output_dim_size: int = None
    max_iterations_for_episode: int = None
    max_do_nothing_offset: int = None
    min_est_error_before_removing_points: float = None
    min_expected_avg_improvement: Optional[float] = None
    time_steps_to_average_improvement: Optional[int] = None
    large_neg_reward: Optional[float] = None


class FlowParamsConfig(BaseModel):
    inflow: str
    mu: str
    rho: float


class GeometryParamsConfig(BaseModel):
    mesh: str


class SolverParamsConfig(BaseModel):
    dt: float
    smooth: bool
    solver_type: str


class GeometryGeneratorParams(BaseModel):
    min_triangles: int
    max_triangles: int
    course_h: float
    fine_h: float

class CorrectPositioningPreTraining(BaseModel):
    num_batches_to_train_for: int
    batch_size: int
    max_reward_for_good_variance: float
    successes_needed_to_switch_to_next_policy: int


class FlowConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    device: torch.device
    agent_params: AgentParamsConfig
    optimizer: OptimizerConfig
    restart_num: int
    restart: bool
    save_dir: str
    epsilon: EpsilonConfig
    flow_params: FlowParamsConfig
    geometry_params: GeometryParamsConfig
    solver_params: SolverParamsConfig
    geom_generator_params: GeometryGeneratorParams
    correct_positioning_pre_training: CorrectPositioningPreTraining

class Transition(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    state: Data
    state_choice_output: torch.Tensor
    next_state: Optional[Data] = None
    reward: torch.Tensor

class BatchedTransition(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    state: List[Data]
    state_choice_output: List[torch.Tensor]
    next_state: List[Union[Data, None]]
    reward: List[torch.Tensor]

class NonRLTransition(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    state: Data
    shape_parameters: ShapeOutlineParameters
    shape_transformation_parameters: ShapeTransformationParameters
    shaplely_polygon: Polygon
    expected_average_point_variance: float
    points: Any


class BatchedNonRLTransition(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    state: List[Data]
    shape_parameters: List[ShapeOutlineParameters]
    shape_transformation_parameters: List[ShapeTransformationParameters]
    shapely_polygon: List[Polygon]
    expected_average_point_variance: List[float]
