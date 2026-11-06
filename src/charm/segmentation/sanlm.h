#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Denoises a column major image in place
void anlm(float *ima, int v, int f, int use_rician, const int *dims);

#ifdef __cplusplus
}
#endif
