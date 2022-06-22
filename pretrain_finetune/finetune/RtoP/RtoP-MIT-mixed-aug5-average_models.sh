onmt_average_models -output  ./exp/USPTO-MIT_RtoP_aug5_mixed/finetuned_average_model_96-100.pt   \
    -m  exp/USPTO-MIT_RtoP_aug5_mixed/finetuned_model.reactants-products_step_960000.pt \
        exp/USPTO-MIT_RtoP_aug5_mixed/finetuned_model.reactants-products_step_970000.pt \
        exp/USPTO-MIT_RtoP_aug5_mixed/finetuned_model.reactants-products_step_980000.pt \
        exp/USPTO-MIT_RtoP_aug5_mixed/finetuned_model.reactants-products_step_990000.pt \
        exp/USPTO-MIT_RtoP_aug5_mixed/finetuned_model.reactants-products_step_1000000.pt