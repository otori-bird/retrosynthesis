onmt_average_models -output  ./exp/USPTO-MIT_PtoR_aug5/finetuned_average_model_46-50.pt \
    -m  exp/USPTO-MIT_PtoR_aug5/finetuned_model.product-reactants_step_460000.pt \
        exp/USPTO-MIT_PtoR_aug5/finetuned_model.product-reactants_step_470000.pt \
        exp/USPTO-MIT_PtoR_aug5/finetuned_model.product-reactants_step_480000.pt \
        exp/USPTO-MIT_PtoR_aug5/finetuned_model.product-reactants_step_490000.pt \
        exp/USPTO-MIT_PtoR_aug5/finetuned_model.product-reactants_step_500000.pt