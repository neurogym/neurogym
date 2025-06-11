import neurogym as ngym

import supervised_train
import RL_train
import train_and_analysis_template as ta


def supervised_all():
    all_envs = ngym.all_envs(tag='supervised')

    # Detection needs to be skipped now because it seems to have an error with dt=100
    # 'ReachingDelayResponse-v0' needs to be skipped now because it has Box action space
    skip_envs = ['Detection-v0', 'ReachingDelayResponse-v0']

    # all_envs = [all_envs[2]]

    # Skipping 'MotorTiming-v0' now because can't make all period same length
    skip_analysis_envs = ['MotorTiming-v0']


    for envid in all_envs:
        if envid in skip_envs:
            continue

        print('Train & analyze env ', envid)

        # supervised_train.train_network(envid)

        if envid in skip_analysis_envs:
            continue

        activity, info, config = supervised_train.run_network(envid)
        ta.analysis_average_activity(activity, info, config)
        ta.analysis_activity_by_condition(activity, info, config)
        ta.analysis_example_units_by_condition(activity, info, config)
        ta.analysis_pca_by_condition(activity, info, config)


envid = 'GoNogo-v0'
print('Train & analyze env ', envid)

RL_train.train_network(envid)
activity, info, config = RL_train.run_network(envid)
ta.analysis_average_activity(activity, info, config)
ta.analysis_activity_by_condition(activity, info, config)
ta.analysis_example_units_by_condition(activity, info, config)
ta.analysis_pca_by_condition(activity, info, config)