#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "itkAffineTransform.h"
#include "itkMultiThreaderBase.h"
#include "itkVersorRigid3DTransform.h"
#include "pyKvlCalculator.h"
#include "pyKvlImage.h"
#include "pyKvlMesh.h"
#include "pyKvlMixture.h"
#include "pyKvlOptimizer.h"
#include "pyKvlTransform.h"
#include "pyKvlRegistration.h"

void setGlobalDefaultNumberOfThreads(int maximumNumberOfThreads){
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads( maximumNumberOfThreads );
}

NB_MODULE(gemsbindings, m) {
    // nanobind aborts the process on exception types that no translator recognizes.
    nb::register_exception_translator([](const std::exception_ptr &exception, void *) {
        try {
            std::rethrow_exception(exception);
        } catch (const itk::ExceptionObject &error) {
            PyErr_SetString(PyExc_RuntimeError, error.what());
        }
    });

    // Property getters default to reference_internal, which ndarray refuses when it already owns its data.
    nb::class_<KvlTransform>(m, "KvlTransform")
            .def(nb::init<const nb::ndarray<const double, nb::shape<4, 4>, nb::f_contig> &>())
            .def_prop_ro("as_numpy_array", &KvlTransform::AsNumpyArray, nb::rv_policy::automatic)
            ;

    nb::class_<KvlImage>(m, "KvlImage")
            .def(nb::init<const std::string &>())
            .def(nb::init<const ImageBuffer &>())
            .def(nb::init<const std::string &, const std::string &>())
            .def_prop_ro("transform_matrix", &KvlImage::GetTransform)
            .def_prop_ro("crop_slices", &KvlImage::GetCropSlices)
            .def("get_image_buffer", &KvlImage::GetImageBuffer)
            .def("write", &KvlImage::Write)
            .def_static("smooth_image_buffer", &KvlImage::smoothImageBuffer)
            ;

    nb::class_<kvl::VoxelSelection>(m, "KvlVoxelSelection")
            .def("__init__", [](kvl::VoxelSelection *self, const std::vector<long> &shape, const nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> &voxels) {
                new (self) kvl::VoxelSelection({shape[0], shape[1], shape[2]}, voxels.data(), voxels.shape(0));
            }, nb::arg("shape"), nb::arg("voxels"))
            ;

    nb::class_<KvlSparsePriors>(m, "KvlSparsePriors")
            .def("compute_posteriors", &KvlSparsePriors::ComputePosteriors, nb::arg("data").noconvert(), nb::arg("means"),
                 nb::arg("inverse_choleskys"), nb::arg("log_scales"), nb::arg("number_of_gaussians_per_class"), nb::arg("keep_posteriors"))
            .def("compute_class_statistics", &KvlSparsePriors::ComputeClassStatistics, nb::arg("data").noconvert(), nb::arg("number_of_classes"))
            .def("compute_bias_field_terms", &KvlSparsePriors::ComputeBiasFieldTerms, nb::arg("masked_images").noconvert(), nb::arg("pair_weights"),
                 nb::arg("weighted_means"), nb::arg("first_contrasts"), nb::arg("second_contrasts"))
            ;

    m.def("scatter_to_volume", &ScatterToVolume, nb::arg("values").noconvert(), nb::arg("column"), nb::arg("voxels").noconvert(), nb::arg("volume").noconvert());
    m.def("subtract_bias_field", &SubtractBiasField, nb::arg("images").noconvert(), nb::arg("column"), nb::arg("row_coefficients"), nb::arg("basis_functions"),
          nb::arg("voxels").noconvert(), nb::arg("data").noconvert());
    m.def("compute_class_likelihoods", &ComputeClassLikelihoods, nb::arg("data").noconvert(), nb::arg("means"), nb::arg("inverse_choleskys"),
          nb::arg("log_scales"), nb::arg("number_of_gaussians_per_class"), nb::arg("likelihoods").noconvert());

    nb::class_<KvlMesh>(m, "KvlMesh")
            .def_prop_rw("points", &KvlMesh::GetPointSet, &KvlMesh::SetPointSet, nb::for_getter(nb::rv_policy::automatic))
            .def_prop_rw("alphas", &KvlMesh::GetAlphas, &KvlMesh::SetAlphas, nb::for_getter(nb::rv_policy::automatic))
            .def("scale", &KvlMesh::Scale)
            .def("rasterize_values", &KvlMesh::RasterizeValues<float>, nb::arg("shape"), nb::arg("values").noconvert())
            .def("rasterize_values", &KvlMesh::RasterizeValues<double>, nb::arg("shape"), nb::arg("values"))
            .def("rasterize", &KvlMesh::RasterizeMesh, nb::arg("shape"), nb::arg("class_number") = -1)
            .def("rasterize_priors", &KvlMesh::RasterizePriors, nb::arg("voxel_selection"))
            .def("segment", &KvlMesh::Segment, nb::arg("voxel_selection"), nb::arg("data").noconvert(), nb::arg("means"), nb::arg("inverse_choleskys"),
                 nb::arg("log_scales"), nb::arg("number_of_gaussians_per_class"), nb::arg("class_fractions"), nb::arg("structure_factors"), nb::arg("labels"))
            // Writes the boundary nodes it adds back into the caller's mask, so it can't work on a converted copy
            .def("get_submesh", &KvlMesh::GetSubmesh, nb::arg("mask").noconvert(), nb::rv_policy::take_ownership)
            ;

    nb::class_<KvlMeshCollection>(m, "KvlMeshCollection")
            .def(nb::init<>())
            .def_prop_rw("k", &KvlMeshCollection::GetK, &KvlMeshCollection::SetK)
            .def_prop_ro("reference_mesh", &KvlMeshCollection::GetReferenceMesh, nb::rv_policy::take_ownership)
            .def("assign", &KvlMeshCollection::Assign, nb::arg("positions"), nb::arg("tetrahedra"), nb::arg("alphas"), nb::arg("can_move"), nb::arg("k"))
            .def("contents", &KvlMeshCollection::Contents)
            .def("transform", &KvlMeshCollection::Transform)
            ;

    nb::class_<KvlCostAndGradientCalculator>(m, "KvlCostAndGradientCalculator")
            .def(nb::init<const std::string &,
                const std::vector<KvlImage *> &,
                const std::string &,
                const KvlTransform *,
                const LikelihoodMatrix &,
                const nb::object &,
                const Matrix &>(),
                nb::arg("type_name"),
                nb::arg("images"),
                nb::arg("boundary_condition"),
                nb::arg("transform").none() = nb::none(),
                nb::arg("likelihoods").none() = nb::none(),
                nb::arg("voxel_selection").none() = nb::none(),
                nb::arg("target_points").none() = nb::none())
            .def("evaluate_mesh_position", &KvlCostAndGradientCalculator::EvaluateMeshPosition)
            .def("evaluate_mesh_cost", &KvlCostAndGradientCalculator::EvaluateMeshCost)
            ;

    nb::class_<KvlOptimizer>(m, "KvlOptimizer")
            .def(nb::init<const std::string &, const KvlMesh &, const KvlCostAndGradientCalculator &, const std::map<std::string, double> &>())
            .def("step_optimizer", &KvlOptimizer::StepOptimizer)
            .def("update_calculator", &KvlOptimizer::UpdateCalculator)
            .def("update_mesh", &KvlOptimizer::UpdateMesh)
            ;

    using RigidRegistration = KvlRegistration<itk::VersorRigid3DTransform<double>>;
    nb::class_<RigidRegistration>(m, "KvlRigidRegistration")
            .def(nb::init<>())
            .def("read_images", &RigidRegistration::ReadImages, nb::arg("fixed_image_file_name"), nb::arg("moving_image_file_name"))
            .def("initialize_transform", &RigidRegistration::InitializeTransform)
            .def("register", &RigidRegistration::Register)
            .def("write_out_result", &RigidRegistration::WriteOutResults, nb::arg("output_file_name"))
            .def("get_transformation_matrix", &RigidRegistration::GetFinalTransformation)
            ;

    using AffineRegistration = KvlRegistration<itk::AffineTransform<double, 3>>;
    nb::class_<AffineRegistration>(m, "KvlAffineRegistration")
            .def(nb::init<double,
                int,
                int,
                const std::vector<double> &,
                double,
                const std::vector<double> &,
                bool,
                double,
                const std::string &>(),
                nb::arg("translation_scale"),
                nb::arg("number_of_iterations"),
                nb::arg("number_of_histogram_bins"),
                nb::arg("shrink_scales"),
                nb::arg("background_gray_level"),
                nb::arg("smoothing_sigma"),
                nb::arg("use_center_of_mass_initialization"),
                nb::arg("sampling_rate"),
                nb::arg("interpolator"))
            .def("read_images", &AffineRegistration::ReadImages, nb::arg("fixed_image_file_name"), nb::arg("moving_image_file_name"))
            .def("initialize_transform", &AffineRegistration::InitializeTransform)
            .def("register", &AffineRegistration::Register)
            .def("write_out_result", &AffineRegistration::WriteOutResults, nb::arg("output_file_name"))
            .def("get_transformation_matrix", &AffineRegistration::GetFinalTransformation)
            ;

    m.def("set_number_of_threads", &setGlobalDefaultNumberOfThreads, "Sets the maximum number of threads for ITK.");
    m.def("get_number_of_threads", []() { return itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads(); });
}
