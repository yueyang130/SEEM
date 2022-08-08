# for level in umaze-v0 umaze-diverse-v0 medium-play-v0 medium-diverse-v0 large-play-v0 large-diverse-v0
# do
#   ./experiments/vis_dist.py --env=antmaze-${level}
# done

# for level in complete-v0 partial-v0 mixed-v0
# do
#   ./experiments/vis_dist.py --env=kitchen-${level}
# done

for scenario in pen hammer door relocate
  do
    for tp in human cloned
    do
      ./experiments/vis_dist.py --env=${scenario}-${tp}-v0
    done
  done