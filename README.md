# CHARM

This project is a port of the CHARM tool from the [simNIBS](https://github.com/simnibs/simnibs) toolkit which has the following major changes:
- Bundled an optimized build of SAMSEG to speedup segmentation
- Optimized numpy code across the entire codebase; replacing several parts with native implementations
- Optimized build of mmg for faster mesh-repair and post-processing
- Optimized CGAL code for faster meshing
- Modernized cyclopts CLI
- Bundled atlases as hdf5 to reduce wheel size
- Distributed as a single stand-alone wheel that doesn't require complicated dependency resolution

These changes put together bring CHARM head-model generation time from ~2 hours to ~6 minutes and peak memory from ~22 Gigs to ~8, all while staying CPU only!

<hr/>

Prebuilt wheels are attached under [releases](https://github.com/vcubiomag/charm/releases) for macOS (arm64) and Linux (x86_64).

<hr/>

Please cite the original authors here:

```
@article{PUONTI2020117044,
  title = {Accurate and robust whole-head segmentation from magnetic resonance images for individualized head modeling},
  journal = {NeuroImage},
  volume = {219},
  pages = {117044},
  year = {2020},
  issn = {1053-8119},
  doi = {https://doi.org/10.1016/j.neuroimage.2020.117044},
  url = {https://www.sciencedirect.com/science/article/pii/S1053811920305309},
  author = {Oula Puonti and Koen {Van Leemput} and Guilherme B. Saturnino and Hartwig R. Siebner and Kristoffer H. Madsen and Axel Thielscher},
}
```
