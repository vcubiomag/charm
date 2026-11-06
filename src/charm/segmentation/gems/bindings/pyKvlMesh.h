#ifndef GEMS_PYKVLMESH_H_H
#define GEMS_PYKVLMESH_H_H

#include <string>
#include <vector>
#include "kvlAtlasMeshCollection.h"
#include "kvlVoxelSelection.h"
#include "pyKvlMixture.h"
#include "pyKvlNumpy.h"
#include "pyKvlTransform.h"

using MeshPointer = kvl::AtlasMesh::ConstPointer;

class KvlMesh {

public:
    NumpyArray<double> GetPointSet() const;
    void SetPointSet(const Matrix &source);
    NumpyArray<double> GetAlphas() const;
    void SetAlphas(const nb::object &source);
    void Scale(const std::vector<double> &scaling);
    NumpyArray<uint16_t> RasterizeMesh(const std::vector<size_t> &size, int classNumber=-1);
    KvlSparsePriors *RasterizePriors(const kvl::VoxelSelection &voxels);
    NumpyArray<uint16_t> Segment(const kvl::VoxelSelection &voxels, const RowMatrix &data, const RowMatrix &means,
                                 const RowTensor &inverseCholeskys, const RowVector &logScales,
                                 const std::vector<int> &numberOfGaussiansPerClass, const RowMatrix &classFractions,
                                 const std::vector<double> &structureFactors, const std::vector<uint16_t> &labels);
    template <typename TValue>
    NumpyArray<TValue> RasterizeValues(std::vector<size_t> size, const nb::ndarray<const TValue, nb::c_contig> &values);
    KvlMesh* GetSubmesh( const nb::ndarray<bool, nb::ndim<1>> &mask );

    explicit KvlMesh(const MeshPointer &aMesh) : mesh(aMesh) {}
    MeshPointer mesh;
};

class KvlMeshCollection {
    kvl::AtlasMeshCollection::Pointer meshCollection = kvl::AtlasMeshCollection::New();
public:
    void Assign(const nb::ndarray<const double, nb::ndim<2>, nb::c_contig> &positions,
                const nb::ndarray<const std::uint32_t, nb::ndim<2>, nb::c_contig> &tetrahedra,
                const nb::ndarray<const float, nb::ndim<2>, nb::c_contig> &alphas,
                const nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig> &canMove, double k);
    nb::tuple Contents() const;
    double GetK() const;
    void SetK(double k);
    KvlMesh* GetReferenceMesh();
    void Transform(const KvlTransform &transform);
};

kvl::AtlasMesh::PointsContainer::Pointer PointsFromNumpy(const Matrix &source);

#endif //GEMS_PYKVLMESH_H_H
