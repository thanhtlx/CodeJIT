# JIT_VulDet_Baselines

# Run VCCFinder
python -m baselines.VCCFinder.vccfinder -split by_time
python -m baselines.VCCFinder.vccfinder -split cross_project

# Run LAPredict
python -m baselines.LApredict.lapredict -split by_time
python -m baselines.LApredict.lapredict -split cross_project

# Run JITLine
by_time: 18 cross_project: 13
python -m baselines.JITLine.jitline -split by_time
python -m baselines.JITLine.jitline -split cross_project