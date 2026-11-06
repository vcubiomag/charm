#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>

#include <cstdlib>
#include <list>
#include <optional>
#include <variant>

#include "cgal.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT FT;
typedef K::Point_3 Point;
typedef K::Segment_3 Segment;
typedef K::Vector_3 Vector;
typedef FT (Function)(const Point&);
typedef CGAL::Surface_mesh<Point> Surface_mesh;

typedef CGAL::AABB_face_graph_triangle_primitive<Surface_mesh> Primitive;
typedef CGAL::AABB_traits_3<K, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
typedef Tree::Intersection_and_primitive_id<Segment>::Type Segment_intersection;
typedef std::optional<Segment_intersection> Segment_intersection_optional;
typedef Tree::Primitive_id Primitive_id;

using namespace CGAL::parameters;

std::pair<std::vector<int>, std::vector<float>> segment_triangle_intersection(
        const float* vertices, int n_vertices, const int* tris, int n_faces,
        const float* segment_start, const float* segment_end, int n_segments)

{
  Surface_mesh m;
  std::vector<Surface_mesh::Vertex_index> vertex_indices(n_vertices);
  for(int i = 0; i < n_vertices; ++i) {
    vertex_indices[i] = m.add_vertex(Point(vertices[3*i], vertices[3*i+1], vertices[3*i+2]));
  }
  for(int i = 0; i < n_faces; ++i) {
    m.add_face(
        vertex_indices[tris[3*i]],
        vertex_indices[tris[3*i+1]],
        vertex_indices[tris[3*i+2]]);
  }
  Tree tree(faces(m).first, faces(m).second, m);
  std::vector<int> indices_pairs;
  std::vector<float> intersect_positions;
  for (int i=0; i < n_segments; i++){
      Segment segment(
        Point(segment_start[3*i], segment_start[3*i + 1], segment_start[3*i + 2]),
        Point(segment_end[3*i], segment_end[3*i + 1], segment_end[3*i + 2])
      );
      std::list<Segment_intersection> intersections;
      tree.all_intersections(segment, std::back_inserter(intersections));
      for (auto it=intersections.begin(); it != intersections.end(); ++it){
          auto& inter = *it;
          std::size_t id = inter.second;
          indices_pairs.push_back(i);
          indices_pairs.push_back((int) id);
          Point* p = std::get_if<Point>(&(inter.first));
          Segment *s = std::get_if<Segment>(&(inter.first));
          if (p) {
              for (int j=0; j<3; j++) intersect_positions.push_back((float) (*p)[j]);
           }
          else if (s) {
              for (int j=0; j<3; j++) intersect_positions.push_back((float) s->source()[j]);
          }
      }
  }
  return std::make_pair(indices_pairs, intersect_positions);
}
