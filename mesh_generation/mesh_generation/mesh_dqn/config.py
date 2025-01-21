"""Config for the mesh generation DQN."""

import os
import yaml
import torch

from mesh_generation.mesh_dqn.pydantic_objects import OptimizerConfig, AgentParamsConfig, FlowConfig, EpsilonConfig, \
    FlowParamsConfig, SolverParamsConfig, GeometryParamsConfig, GeometryGeneratorParams, CorrectPositioningPreTraining


def load_config(restart: bool, prefix: str) -> FlowConfig:
    # Load config
    prefix += "_"
    save_dir = 'training_results/' + prefix[:-1]
    RESTART_NUM = 0

    print(f"./{save_dir}/config.yaml")
    with open(f"./{save_dir}/config.yaml", 'r') as stream:
        flow_config = yaml.safe_load(stream)
    print(f"\n\nrestart NUM: {RESTART_NUM}\n\n")

    flow_config['agent_params']['plot_dir'] = save_dir
    flow_config['agent_params']['prefix'] = prefix
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
    flow_config['optimizer']['lr'] = 1e-4
    flow_config['optimizer']['weight_decay'] = float(flow_config['optimizer']['weight_decay'])
    flow_config['optimizer']['batch_size'] = 10

    # Convert agent_params values to the correct types
    flow_config['agent_params']['target_update'] = int(flow_config['agent_params']['target_update'])
    flow_config['agent_params']['num_workers'] = int(flow_config['agent_params']['num_workers'])
    flow_config['agent_params']['num_parallel'] = int(flow_config['agent_params']['num_parallel'])

    # flow_config['agent_params']['large_neg_reward'] = 1

    # output dims:
    # 0: terminate
    # 1: add node suggestion strength
    # 2: remove node suggestion strength
    # 3: both add and remove node suggestion strength
    # 4: node to add x
    # 5: node to add y
    # 6: node to remove x
    # 7: node to remove y
    flow_config['agent_params']['output_dim_size'] = 8

    flow_config['agent_params']['max_iterations_for_episode'] = 200

    flow_config['agent_params']['max_do_nothing_offset'] = 10

    flow_config['agent_params']['expected_avg_improvement'] = 0.0004

    flow_config['agent_params']['min_est_error_before_removing_points'] = 0.000014

    flow_config['agent_params']['min_expected_avg_improvement'] = 0.0002
    flow_config['agent_params']['time_steps_to_average_improvement'] = 10

    geometry_generator_params = GeometryGeneratorParams(
        min_triangles=1,
        max_triangles=1,
        course_h=0.1,
        fine_h=0.01
    )

    correct_position_training = CorrectPositioningPreTraining(
        num_batches_to_train_for=10000,
        batch_size=40,
        max_reward_for_good_variance=0.3,
        successes_needed_to_switch_to_next_policy = 20
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
        geom_generator_params=geometry_generator_params,
        correct_positioning_pre_training=correct_position_training
    )
