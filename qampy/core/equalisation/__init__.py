from qampy.core.equalisation.equalisation import equalise_signal, dual_mode_equalisation, apply_filter, DATA_AIDED, \
    REAL_VALUED, DECISION_BASED, NONDECISION_BASED, TRAINING_FCTS, __doc__

# the below is a hack to make sphinx pick up the documentation

#: Decision based equalisation methods
DECISION_BASED = DECISION_BASED

#: Non-decision based equalisation methods
NONDECISION_BASED = NONDECISION_BASED

#: Real-valued equalisation methods
REAL_VALUED = REAL_VALUED

#: Data-aided equalisation methods
DATA_AIDED = DATA_AIDED

#: All available equaliser method#:
TRAINING_FCTS = TRAINING_FCTS