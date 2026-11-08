# CHARM

This project is a port of the CHARM tool from the [simNIBS](https://github.com/simnibs/simnibs) toolkit which has the following major changes:
- Bundled an optimized build of SAMSEG to speedup segmentation
- Optimized numpy code across the entire codebase; replacing several parts with native implementations
- Optimized build of mmg for faster mesh-repair and post-processing
- Optimized CGAL code for faster meshing
- Modernized cyclopts CLI
- Bundled atlases as hdf5 to reduce wheel size
- Distributed as a single stand-alone wheel that doesn't require complicated dependency resolution

These changes put together bring CHARM head-model generation time from ~2 hours to ~6 minutes while staying CPU only.

<hr/>

Please cite the original authors here:

Puonti O, Van Leemput K, Saturnino GB, Siebner HR, Madsen KH, Thielscher A. (2020). Accurate and robust whole-head segmentation from magnetic resonance images for individualized head modeling. Neuroimage, [219:117044](https://doi.org/10.1016/j.neuroimage.2020.117044).
