for adv_fn in mean max
do
  for crr_fn in exp indicator
  do
    for adv_norm in True False
    do
      ADV_FN=$adv_fn CRR_FN=$crr_fn ADV_NORM=$adv_norm ./scripts/debug_dql.sh
    done
  done
done
