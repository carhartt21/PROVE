#!/bin/bash
# Submit all domain adaptation test jobs

cd ${HOME}/repositories/PROVE/jobs/domain_adaptation_missing

bsub < da_baseline_bdd10k_pspnet_r50.sh
bsub < da_baseline_bdd10k_segformer_mit-b5.sh
bsub < da_baseline_idd-aw_pspnet_r50.sh
bsub < da_baseline_idd-aw_segformer_mit-b5.sh
bsub < da_gen_step1x_new_idd-aw_pspnet_r50.sh
bsub < da_std_autoaugment_idd-aw_pspnet_r50.sh
bsub < da_std_autoaugment_idd-aw_segformer_mit-b5.sh
bsub < da_std_cutmix_idd-aw_pspnet_r50.sh
bsub < da_std_cutmix_idd-aw_segformer_mit-b5.sh
bsub < da_std_mixup_idd-aw_pspnet_r50.sh
bsub < da_std_mixup_idd-aw_segformer_mit-b5.sh
bsub < da_std_randaugment_idd-aw_pspnet_r50.sh

echo 'Submitted 12 jobs'
