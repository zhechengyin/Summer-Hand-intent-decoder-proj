"""Neural Intent Decoder -- a two-stage (N1 + N2) motor-imagery decoding pipeline.

Package layout
--------------
config.py            load config.yaml, seeding, path helpers
containers.py        TrialEpochs -- the array container passed between stages
load_bids.py         discover BIDS files, load EEG (.set) and fNIRS (.mat)
preprocess_eeg.py    filter / notch / epoch EEG around the imagery window
preprocess_fnirs.py  optical density -> Beer-Lambert HbO/HbR -> epoch
feature_extraction.py bandpower (EEG) and hemodynamic (fNIRS) features
fusion.py            trial-aligned feature-level fusion + sklearn pipelines
train_n1.py          N1 neural-evidence decoder (probability output)
evaluate.py          subject-specific + leave-one-subject-out evaluation
state.py             ProstheticState -- the avatar/prosthetic state
mini_ai_spine_n2.py  N2 state-injected intent-to-command interpreter
simulate_avatar.py   apply commands to state + render a simple arm
replay_time_domain.py the online-style N1 -> N2 -> avatar replay loop
synthetic.py         class-structured synthetic epochs for smoke testing
"""

__version__ = "0.1.0"
