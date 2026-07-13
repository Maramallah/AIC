# Distillation Notes

Distillation is to check it will reproduce similar checkpoint.
but For final inference, it's recommended to use:
checkpoints/best_checkpoint.pth.tar you must download it from releases Because it is proven and tested on test data kaggle leaderboard.


Distillation requires:
- training data
- teacher predictions
- base/pre-distillation LightFC checkpoint

note: A newly distilled checkpoint most probably will produce different CSV outputs from the released checkpoint.
