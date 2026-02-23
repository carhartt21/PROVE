#!/bin/bash
# Submit all domain adaptation retest jobs

bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_baseline_bdd10k_pspnet_r50.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_baseline_bdd10k_segformer_mit_b5.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_baseline_idd_aw_pspnet_r50.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_baseline_idd_aw_segformer_mit_b5.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_gen_step1x_new_idd_aw_pspnet_r50.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_std_autoaugment_idd_aw_pspnet_r50.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_std_autoaugment_idd_aw_segformer_mit_b5.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_std_cutmix_idd_aw_pspnet_r50.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_std_cutmix_idd_aw_segformer_mit_b5.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_std_mixup_idd_aw_pspnet_r50.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_std_mixup_idd_aw_segformer_mit_b5.sh
bsub < ${HOME}/repositories/PROVE/jobs/domain_adaptation_retest/da2_std_randaugment_idd_aw_pspnet_r50.sh
