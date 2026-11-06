#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Image_3.h>

#include <CGAL/Random.h>
#include "tbb/task_arena.h"
#include "tbb/task_group.h"

#include <cstdlib>

#include "cgal.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT FT;
typedef K::Point_3 Point;
typedef K::Segment_3 Segment;
typedef K::Vector_3 Vector;
typedef FT (Function)(const Point&);
typedef CGAL::Labeled_mesh_domain_3<K> Mesh_domain_img;
typedef CGAL::Parallel_tag Concurrency_tag;
typedef CGAL::Mesh_triangulation_3<Mesh_domain_img,CGAL::Default,Concurrency_tag>::type Tr_img;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr_img> C3t3_img;
typedef CGAL::Mesh_criteria_3<Tr_img> Mesh_criteria_img;
typedef Mesh_criteria_img::Facet_criteria    Facet_criteria_img;
typedef Mesh_criteria_img::Cell_criteria     Cell_criteria_img;

namespace params = CGAL::parameters;


int mesh_image(
  const char *fn_image,
  const char *fn_out,
  float facet_angle,
  float facet_size,
  float facet_distance,
  float cell_radius_edge_ratio,
  float cell_size,
  bool do_perturb,
  bool do_exude,
  bool do_lloyd
)
{
  CGAL::Image_3 image;
  if(!image.read(fn_image)){
    std::cerr << "Error: Cannot read file " <<  fn_image << std::endl;
    return EXIT_FAILURE;
  }

  CGAL::get_default_random() = CGAL::Random(0);

  Mesh_domain_img domain = Mesh_domain_img::create_labeled_image_mesh_domain(image, params::relative_error_bound(1e-10));

  Facet_criteria_img facet_criteria(facet_angle, facet_size, facet_distance);
  Cell_criteria_img cell_criteria(cell_radius_edge_ratio, cell_size);
  Mesh_criteria_img criteria(facet_criteria, cell_criteria);

  C3t3_img c3t3 = CGAL::make_mesh_3<C3t3_img>(domain, criteria, params::no_perturb(), params::no_exude());

  if (do_lloyd){
    // Lloyd often fails in parallel: https://github.com/CGAL/cgal/issues/4566
    tbb::task_arena limited(1);
    tbb::task_group tg;
    limited.execute([&]{ tg.run([&]{ CGAL::lloyd_optimize_mesh_3(c3t3, domain); }); });
    limited.execute([&]{ tg.wait(); });
  }
  if (do_perturb) CGAL::perturb_mesh_3(c3t3, domain);
  if (do_exude) CGAL::exude_mesh_3(c3t3);

  std::ofstream medit_file(fn_out);
  c3t3.output_to_medit(medit_file);

  return EXIT_SUCCESS;
}

struct Sizing_field
{
    typedef ::FT FT;
    typedef Point Point_3;
    typedef Mesh_domain_img::Index Index;
    double tx, ty, tz;
    double vx, vy, vz;
    std::size_t sx, sy, sz;
    const float *sizing_field_image;

    FT operator()(const Point_3 &p, const int, const Index&) const
    {
        // CGAL assumes centered voxels
        const std::size_t i = (p.x() - tx)/vx + 0.5;
        const std::size_t j = (p.y() - ty)/vy + 0.5;
        const std::size_t k = (p.z() - tz)/vz + 0.5;
        if (i < 0 || j < 0 || k < 0 || i >= sx || j >= sy || k >= sz) {
            std::cerr << "trying to access sizing field out-of-bounds" << std::endl;
            return 1;
        }
	      FT val = sizing_field_image[i + sx*j + sx*sy*k];
        return val;
    };
};

int mesh_image_sizing_field(
  const char *fn_image,
  const char *fn_out,
  float facet_angle,
  const float *facet_size,
  const float *facet_distance,
  float cell_radius_edge_ratio,
  const float *cell_size,
  bool do_perturb,
  bool do_exude,
  bool do_lloyd
)
{
  CGAL::Image_3 image;
  if(!image.read(fn_image)){
    std::cerr << "Error: Cannot read file " <<  fn_image << std::endl;
    return EXIT_FAILURE;
  }

  CGAL::get_default_random() = CGAL::Random(0);

  Mesh_domain_img domain = Mesh_domain_img::create_labeled_image_mesh_domain(image, params::relative_error_bound(1e-10));

  Sizing_field sizing_field_cell = {
     image.tx(), image.ty(), image.tz(),
     image.vx(), image.vy(), image.vz(),
     image.xdim(), image.ydim(), image.zdim(),
     cell_size
  };
  Sizing_field sizing_field_facet_distance = {
     image.tx(), image.ty(), image.tz(),
     image.vx(), image.vy(), image.vz(),
     image.xdim(), image.ydim(), image.zdim(),
     facet_distance
  };
  Sizing_field sizing_field_facet_size = {
     image.tx(), image.ty(), image.tz(),
     image.vx(), image.vy(), image.vz(),
     image.xdim(), image.ydim(), image.zdim(),
     facet_size
  };
  Facet_criteria_img facet_criteria(
          facet_angle,
          sizing_field_facet_size,
          sizing_field_facet_distance
  );
  Cell_criteria_img cell_criteria(
          cell_radius_edge_ratio,
          sizing_field_cell
  );
  Mesh_criteria_img criteria(facet_criteria, cell_criteria);

  C3t3_img c3t3 = CGAL::make_mesh_3<C3t3_img>(domain, criteria, params::no_perturb(), params::no_exude());

  if (do_lloyd){
    // Lloyd often fails in parallel: https://github.com/CGAL/cgal/issues/4566
    tbb::task_arena limited(1);
    tbb::task_group tg;
    limited.execute([&]{ tg.run([&]{ CGAL::lloyd_optimize_mesh_3(c3t3, domain); }); });
    limited.execute([&]{ tg.wait(); });
  }
  if (do_perturb) CGAL::perturb_mesh_3(c3t3, domain);
  if (do_exude) CGAL::exude_mesh_3(c3t3);

  std::ofstream medit_file(fn_out);
  c3t3.output_to_medit(medit_file);

  return EXIT_SUCCESS;
}
