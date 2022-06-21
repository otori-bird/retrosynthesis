onmt_average_models -output  ./exp/USPTO-MIT_RtoP_aug5_separated/finetuned_average_model_46-50.pt   \
    -m  exp/USPTO-MIT_RtoP_aug5_separated/finetuned_model.reactants-products_step_460000.pt \
        exp/USPTO-MIT_RtoP_aug5_separated/finetuned_model.reactants-products_step_470000.pt \
        exp/USPTO-MIT_RtoP_aug5_separated/finetuned_model.reactants-products_step_480000.pt \
        exp/USPTO-MIT_RtoP_aug5_separated/finetuned_model.reactants-products_step_490000.pt \
        exp/USPTO-MIT_RtoP_aug5_separated/finetuned_model.reactants-products_step_500000.pt