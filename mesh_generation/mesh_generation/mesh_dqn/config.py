"""Config for the mesh generation DQN."""

import os
import yaml
import torch

from mesh_generation.mesh_dqn.pydantic_objects import OptimizerConfig, AgentParamsConfig, FlowConfig, EpsilonConfig, \
    FlowParamsConfig, SolverParamsConfig, GeometryParamsConfig, GeometryGeneratorParams


def load_config(restart: bool) -> FlowConfig:
    # Load config
    PREFIX = 'ys930_results_'
    save_dir = 'training_results/' + PREFIX[:-1]
    RESTART_NUM = 0

    if restart:
        print(f"./{save_dir}/config.yaml")
        with open(f"./{save_dir}/config.yaml", 'r') as stream:
            flow_config = yaml.safe_load(stream)
        for f in os.listdir(save_dir):
            RESTART_NUM += int(f"{PREFIX}policy_net_1.pt" in f)
        print(f"\n\nrestart NUM: {RESTART_NUM}\n\n")
    else:
        with open(f"./configs/ray_{PREFIX.split('_')[0]}.yaml", 'r') as stream:
            flow_config = yaml.safe_load(stream)

    if not restart:
        with open(f"{save_dir}/config.yaml", 'w') as fout:
            yaml.dump(flow_config, fout)

    flow_config['agent_params']['plot_dir'] = save_dir
    flow_config['agent_params']['prefix'] = PREFIX
    flow_config['restart_num'] = RESTART_NUM
    flow_config['restart'] = restart
    flow_config['save_dir'] = save_dir

    NUM_INPUTS = 13
    flow_config['agent_params']['NUM_INPUTS'] = NUM_INPUTS

    if torch.cuda.is_available():
        print("USING GPU")
        device = torch.device('cuda:0')
    else:
        print("USING CPU")
        device = torch.device('cpu')
    flow_config['device'] = device

    # Convert optimizer values to the correct types
    flow_config['optimizer']['lr'] = float(flow_config['optimizer']['lr'])
    flow_config['optimizer']['weight_decay'] = float(flow_config['optimizer']['weight_decay'])
    flow_config['optimizer']['batch_size'] = int(flow_config['optimizer']['batch_size'])

    # Convert agent_params values to the correct types
    flow_config['agent_params']['target_update'] = int(flow_config['agent_params']['target_update'])
    flow_config['agent_params']['num_workers'] = int(flow_config['agent_params']['num_workers'])
    flow_config['agent_params']['num_parallel'] = int(flow_config['agent_params']['num_parallel'])

    flow_config['agent_params']['output_dim_size'] = 5

    flow_config['agent_params']['max_iterations_for_episode'] = 200

    flow_config['agent_params']['max_do_nothing_offset'] = 10

    geometry_generator_params = GeometryGeneratorParams(
        min_triangles=1,
        max_triangles=5,
        course_h=0.15,
        fine_h=0.01
    )

    # Use Pydantic to validate and create a FlowConfig object
    return FlowConfig(
        agent_params=AgentParamsConfig(**flow_config['agent_params']),
        optimizer=OptimizerConfig(**flow_config['optimizer']),
        restart_num=flow_config['restart_num'],
        restart=flow_config['restart'],
        save_dir=flow_config['save_dir'],
        device=flow_config['device'],
        epsilon=EpsilonConfig(**flow_config['epsilon']),
        flow_params=FlowParamsConfig(**flow_config['flow_config']['flow_params']),
        geometry_params=GeometryParamsConfig(**flow_config['flow_config']['geometry_params']),
        solver_params=SolverParamsConfig(**flow_config['flow_config']['solver_params']),
        geom_generator_params=geometry_generator_params
    )
