onmt_average_models -output  ./exp/USPTO_50K_P2S2R_aug20/S2R/average_model_96-100.pt \
    -m  exp/USPTO_50K_P2S2R_aug20/S2R/model.synthons-reactants_step_960000.pt \
        exp/USPTO_50K_P2S2R_aug20/S2R/model.synthons-reactants_step_970000.pt \
        exp/USPTO_50K_P2S2R_aug20/S2R/model.synthons-reactants_step_980000.pt \
        exp/USPTO_50K_P2S2R_aug20/S2R/model.synthons-reactants_step_9900000.pt \
        exp/USPTO_50K_P2S2R_aug20/S2R/model.synthons-reactants_step_1000000.pt
