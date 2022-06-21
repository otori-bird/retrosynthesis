onmt_average_models -output  ./exp/USPTO_50K_PtoR_aug20/finetune_average_model_26-30.pt \
    -m  exp/USPTO_50K_PtoR_aug20/finetune_model.product-reactants_step_260000.pt \
        exp/USPTO_50K_PtoR_aug20/finetune_model.product-reactants_step_270000.pt \
        exp/USPTO_50K_PtoR_aug20/finetune_model.product-reactants_step_280000.pt \
        exp/USPTO_50K_PtoR_aug20/finetune_model.product-reactants_step_290000.pt \
        exp/USPTO_50K_PtoR_aug20/finetune_model.product-reactants_step_300000.pt
