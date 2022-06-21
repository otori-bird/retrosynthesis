onmt_average_models -output  ./exp/USPTO_50K_P2S2R_aug20/P2S/average_model_96-100.pt \
    -m  exp/USPTO_50K_P2S2R_aug20/P2S/model.product-synthons_step_960000.pt \
        exp/USPTO_50K_P2S2R_aug20/P2S/model.product-synthons_step_970000.pt \
        exp/USPTO_50K_P2S2R_aug20/P2S/model.product-synthons_step_980000.pt \
        exp/USPTO_50K_P2S2R_aug20/P2S/model.product-synthons_step_990000.pt \
        exp/USPTO_50K_P2S2R_aug20/P2S/model.product-synthons_step_1000000.pt
