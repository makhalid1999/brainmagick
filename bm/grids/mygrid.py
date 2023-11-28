from ._explorers import ClipExplorer


# Decorator class defines what metrics to track and other metadata.
@ClipExplorer
def explorer(launcher):
	# Methods `slurm_` and `bind_` are in-place.
    launcher.slurm_(gpus=4)  # All XPs scheduled with `launcher` will use 4 gpus.
    launcher.bind_({'norm.max_scale': 20, 'dset.n_recordings': 2})  # set common params.

    more_subjects = {'dset.n_recordings': 100}
    selections = ['brennan2019']

    for selection in selections:
        # The `bind()` method returns a sub-launcher with different params.
        # This won't affect the original launcher.
        sub = launcher.bind({'dset.selections': [selection]})
        # You schedule experiments by calling the (sub)-launcher.
        sub()
