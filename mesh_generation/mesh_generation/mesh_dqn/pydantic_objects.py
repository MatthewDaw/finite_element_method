from typing import Optional
from pydantic import BaseModel
import torch


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
    time_reward: Optional[float] = None
    timesteps: Optional[int] = None
    u: Optional[float] = None
    output_dim_size: int = None
    max_iterations_for_episode: int = None
    max_do_nothing_offset: int = None


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
