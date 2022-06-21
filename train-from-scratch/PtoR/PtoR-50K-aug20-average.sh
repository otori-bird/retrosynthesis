onmt_average_models -output  ./exp/USPTO_50K_PtoR_aug20/average_model_56-60.pt \
    -m  exp/USPTO_50K_PtoR_aug20/model.product-reactants_step_560000.pt \
        exp/USPTO_50K_PtoR_aug20/model.product-reactants_step_570000.pt \
        exp/USPTO_50K_PtoR_aug20/model.product-reactants_step_580000.pt \
        exp/USPTO_50K_PtoR_aug20/model.product-reactants_step_590000.pt \
        exp/USPTO_50K_PtoR_aug20/model.product-reactants_step_600000.pt
