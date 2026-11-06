#pragma once

#include <utility>
#include <vector>

int mesh_image(const char *fn_image, const char *fn_out, float facet_angle, float facet_size, float facet_distance,
               float cell_radius_edge_ratio, float cell_size, bool do_perturb, bool do_exude, bool do_lloyd);

// The sizing fields are Fortran ordered images on the grid of the input image
int mesh_image_sizing_field(const char *fn_image, const char *fn_out, float facet_angle, const float *facet_size,
                            const float *facet_distance, float cell_radius_edge_ratio, const float *cell_size,
                            bool do_perturb, bool do_exude, bool do_lloyd);

// For each intersection, the segment and triangle as consecutive entries of the first vector, and its position as
// three entries of the second
std::pair<std::vector<int>, std::vector<float>> segment_triangle_intersection(const float *vertices, int n_vertices,
                                                                              const int *tris, int n_faces,
                                                                              const float *segment_start,
                                                                              const float *segment_end, int n_segments);
