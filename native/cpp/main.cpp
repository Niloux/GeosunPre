#include <cblas.h>
#include <ceres/ceres.h>
#include <colmap/controllers/incremental_pipeline.h>
#include <colmap/estimators/bundle_adjustment_ceres.h>
#include <colmap/exe/sfm.h>
#include <colmap/scene/reconstruction_manager.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "delivery.h"
#include "depth_cost.h"
#include "las_io.h"
#include "pipeline.h"
#include "registration.h"
#include "runtime.h"
#include "scene.h"
#include "sfm_pipeline.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

class BAProgress final : public ceres::IterationCallback {
 public:
  explicit BAProgress(std::string stage) : stage_(std::move(stage)) {}
  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& step) override {
    if (Seconds(updated_) >= 30.) {
      std::cerr << "Native BA " << stage_ << ": iteration " << step.iteration
                << ", cost " << step.cost << ", elapsed " << Seconds(started_)
                << "s\n";
      updated_ = Clock::now();
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  std::string stage_;
  Clock::time_point started_ = Clock::now(), updated_ = started_;
};

Json Environment() {
  return {{"implementation", 1},
          {"colmap", "4.2.0"},
          {"ceres", CERES_VERSION_STRING},
          {"blas_config", openblas_get_config()},
          {"blas_cpu", openblas_get_corename()},
          {"blas_threads", openblas_get_num_threads()},
          {"runtime", "cpp"},
          {"ceres_cuda_dense",
           ceres::IsDenseLinearAlgebraLibraryTypeAvailable(ceres::CUDA)}};
}

Json Cheirality(const colmap::Reconstruction& rec) {
  uint64_t total = 0, positive = 0;
  for (const auto& [id, image] : rec.Images()) {
    if (!image.HasPose()) continue;
    const auto pose = image.CamFromWorld();
    for (const auto& obs : image.Points2D()) {
      if (!obs.HasPoint3D()) continue;
      const Eigen::Vector3d local = pose * rec.Point3D(obs.point3D_id).xyz;
      ++total;
      if (local.allFinite() && local.z() > 1e-9) ++positive;
    }
  }
  return {{"observations", total},
          {"positive_depth", positive},
          {"invalid_depth", total - positive}};
}

Json CameraHealth(const colmap::Reconstruction& rec, const Json& initial,
                  bool free) {
  Json health = Json::object();
  for (const auto& [cid, camera] : rec.Cameras()) {
    const auto key = std::to_string(cid);
    const auto base = initial.at(key).get<std::vector<double>>();
    const auto& p = camera.params;
    if (camera.ModelName() != "OPENCV_FISHEYE" || p.size() != 8 ||
        base.size() != 8)
      throw std::runtime_error("Expected shared OPENCV_FISHEYE intrinsics");
    for (double v : p)
      if (!std::isfinite(v))
        throw std::runtime_error("Nonfinite camera parameters");
    for (int j = 0; j < 256; ++j) {
      const double theta = (80. * 3.14159265358979323846 / 180.) * j / 255.;
      double derivative = 1.;
      for (int k = 0; k < 4; ++k)
        derivative += (2 * k + 3) * p[4 + k] * std::pow(theta, 2 * k + 2);
      if (derivative <= 0)
        throw std::runtime_error("Nonmonotonic fisheye calibration");
    }
    const std::vector<double> delta = {base[0] * .1, base[1] * .1, 50,  50,
                                       .05,          .02,          .01, .005};
    Json difference = Json::array(), hit = Json::array();
    for (size_t j = 0; j < 8; ++j) {
      difference.push_back(p[j] - base[j]);
      if (free && std::min(p[j] - (base[j] - delta[j]),
                           base[j] + delta[j] - p[j]) < 1e-4 * 2 * delta[j])
        hit.push_back(j);
    }
    health[key] = {{"params", p},
                   {"delta", difference},
                   {"monotonic_in_160deg", true},
                   {"parameters_at_bounds", hit}};
  }
  return health;
}

Json SolveBA(colmap::Reconstruction& rec, const Arguments& args,
             const Json& initial) {
  const auto stage = args.Get("--stage", "fixed");
  if (stage != "fixed" && stage != "focal_low_radial" &&
      stage != "all_radial" && stage != "principal")
    throw std::invalid_argument("Unknown BA stage: " + stage);
  const bool free = stage != "fixed";
  const bool guarded =
      stage == "principal" || args.Get("--protect-depth") == "true";
  std::cerr << "Native BA " << stage << ": preparing " << rec.NumRegImages()
            << " images, " << rec.ComputeNumObservations() << " observations\n";
  const auto build_started = Clock::now();
  colmap::BundleAdjustmentOptions options;
  options.refine_focal_length = free;
  options.refine_extra_params = free;
  options.refine_principal_point = stage == "principal";
  options.refine_sensor_from_rig = false;
  options.print_summary = false;
  options.ceres->loss_function_type =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType::HUBER;
  options.ceres->loss_function_scale = 2.;
  auto& solver = options.ceres->solver_options;
  solver.num_threads = args.Integer("--workers", 4);
  solver.max_num_iterations = args.Integer("--iterations", guarded ? 400 : 100);
  solver.function_tolerance = 1e-7;
  solver.minimizer_progress_to_stdout = args.Get("--progress") == "true";
  if (guarded) {
    solver.initial_trust_region_radius = 10.;
    solver.max_trust_region_radius = 10000.;
  }
  const bool gpu = args.Get("--gpu") == "true";
  if (gpu) {
    if (!ceres::IsDenseLinearAlgebraLibraryTypeAvailable(ceres::CUDA))
      throw std::runtime_error("Ceres CUDA solver is unavailable");
    options.ceres->auto_select_solver_type = false;
    solver.linear_solver_type = ceres::DENSE_SCHUR;
    solver.dense_linear_algebra_library_type = ceres::CUDA;
  }
  colmap::BundleAdjustmentConfig config;
  for (const auto id : rec.RegImageIds()) config.AddImage(id);
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  auto adjuster =
      colmap::CreateDefaultCeresBundleAdjuster(options, config, rec);
  auto& problem = *adjuster->Problem();
  struct Guard {
    const geosunpre::PositiveDepthCost* cost;
    const double* point;
    const double* pose;
  };
  std::vector<Guard> guards;
  if (guarded) {
    guards.reserve(rec.ComputeNumObservations());
    for (const auto& [id, image] : rec.Images()) {
      if (!image.HasPose()) continue;
      auto& frame = rec.Frame(image.FrameId());
      size_t image_count = 0;
      for (const auto& data : frame.ImageIds()) ++image_count;
      if (image_count != 1)
        throw std::runtime_error("Expected independent exposure frames");
      auto& pose = frame.RigFromWorld();
      const bool was_variable = problem.HasParameterBlock(pose.params.data());
      for (const auto& obs : image.Points2D()) {
        if (!obs.HasPoint3D()) continue;
        auto& point = rec.Point3D(obs.point3D_id);
        const double depth = (pose * point.xyz).z();
        auto* cost = new geosunpre::PositiveDepthCost(depth);
        problem.AddResidualBlock(cost, nullptr, point.xyz.data(),
                                 pose.params.data());
        guards.push_back({cost, point.xyz.data(), pose.params.data()});
      }
      if (!was_variable) problem.SetParameterBlockConstant(pose.params.data());
    }
  }
  if (free) {
    for (const auto& [cid, source_camera] : rec.Cameras()) {
      auto& camera = rec.Camera(cid);
      const auto base =
          initial.at(std::to_string(cid)).get<std::vector<double>>();
      if (base.size() != 8 || camera.params.size() != 8)
        throw std::runtime_error("Expected 8 fisheye parameters");
      const std::vector<double> delta = {base[0] * .1, base[1] * .1, 50,  50,
                                         .05,          .02,          .01, .005};
      for (int j = 0; j < 8; ++j) {
        problem.SetParameterLowerBound(camera.params.data(), j,
                                       base[j] - delta[j]);
        problem.SetParameterUpperBound(camera.params.data(), j,
                                       base[j] + delta[j]);
      }
      std::vector<int> constant;
      if (stage == "focal_low_radial")
        constant = {2, 3, 6, 7};
      else if (stage == "all_radial")
        constant = {2, 3};
      problem.SetManifold(camera.params.data(),
                          new ceres::SubsetManifold(8, constant));
    }
  }
  const double build_seconds = Seconds(build_started);
  const auto solve_started = Clock::now();
  BAProgress progress(stage);
  solver.callbacks.push_back(&progress);
  std::cerr << "Native BA " << stage << ": solving, preparation "
            << build_seconds << "s\n";
  ceres::Solver::Summary solved;
  ceres::Solve(options.ceres->CreateSolverOptions(config, problem), &problem,
               &solved);
  auto summary =
      colmap::CeresBundleAdjustmentSummary::Create(std::move(solved));
  if (!summary->IsSolutionUsable())
    throw std::runtime_error(summary->BriefReport() + "\n" +
                             summary->ceres_summary.FullReport());
  rec.UpdatePoint3DErrors();
  const auto health_started = Clock::now();
  auto depth = Cheirality(rec);
  if (depth.at("invalid_depth").get<uint64_t>() != 0)
    throw std::runtime_error("BA produced nonphysical depths");
  uint64_t active = 0;
  double barrier_cost = 0.;
  for (const auto& guard : guards) {
    double residual;
    const double* parameters[] = {guard.point, guard.pose};
    guard.cost->Evaluate(parameters, &residual, nullptr);
    if (residual > 0) ++active;
    barrier_cost += .5 * residual * residual;
  }
  auto camera_health = CameraHealth(rec, initial, free);
  std::ostringstream termination;
  termination << summary->termination_type;
  const auto& s = summary->ceres_summary;
  std::cerr << "Native BA " << stage << ": " << summary->BriefReport()
            << ", elapsed " << Seconds(solve_started) << "s\n";
  return {{"stage", stage},
          {"elapsed_seconds", Seconds(solve_started)},
          {"build_seconds", build_seconds},
          {"health_seconds", Seconds(health_started)},
          {"solver", summary->BriefReport()},
          {"solver_details",
           {{"message", s.message},
            {"initial_cost", s.initial_cost},
            {"final_cost", s.final_cost},
            {"successful_steps", s.num_successful_steps},
            {"unsuccessful_steps", s.num_unsuccessful_steps},
            {"full_report", s.FullReport()}}},
          {"cheirality", depth},
          {"positive_depth_barriers", guards.size()},
          {"depth_backend", guarded ? Json("geosunpre_cpp") : Json(nullptr)},
          {"solver_threads", solver.num_threads},
          {"positive_depth_barrier_health",
           {{"active", active}, {"cost", barrier_cost}}},
          {"termination_type",
           "BundleAdjustmentTerminationType." + termination.str()},
          {"max_iterations", solver.max_num_iterations},
          {"initial_trust_region_radius", solver.initial_trust_region_radius},
          {"max_trust_region_radius", solver.max_trust_region_radius},
          {"cameras", camera_health},
          {"environment", Environment()},
          {"gpu", gpu}};
}

void RunBA(const Arguments& args) {
  args.Validate({"--input", "--output", "--initial-model", "--initial-cameras",
                 "--stage", "--workers", "--iterations", "--protect-depth",
                 "--gpu", "--progress", "--blas-threads"});
  const auto input = args.Path("--input"), output = args.Path("--output");
  if (fs::exists(output))
    throw std::runtime_error("Output already exists: " + output.u8string());
  const auto temporary =
      fs::path(output.native() + fs::path(".pending").native());
  if (fs::exists(temporary))
    throw std::runtime_error("Unfinished output exists: " +
                             temporary.u8string());
  const auto started = Clock::now();
  colmap::Reconstruction rec;
  rec.Read(input);
  if (!rec.IsValid())
    throw std::runtime_error("Invalid BA input reconstruction");
  Json initial;
  if (!args.Get("--initial-cameras").empty())
    initial = ReadJson(args.Path("--initial-cameras"));
  else {
    colmap::Reconstruction reference;
    reference.Read(args.Path("--initial-model"));
    for (const auto& [id, camera] : reference.Cameras())
      initial[std::to_string(id)] = camera.params;
  }
  auto report = SolveBA(rec, args, initial);
  if (!rec.IsValid()) throw std::runtime_error("Invalid solved reconstruction");
  fs::create_directories(temporary / "sparse/0");
  rec.Write(temporary / "sparse/0");
  colmap::Reconstruction reread;
  reread.Read(temporary / "sparse/0");
  if (!reread.IsValid())
    throw std::runtime_error("Invalid serialized BA result");
  report["registered"] = rec.NumRegImages();
  report["points"] = rec.NumPoints3D();
  report["observations"] = rec.ComputeNumObservations();
  report["wall_seconds"] = Seconds(started);
  report["resources"] = Resources();
  WriteJson(temporary / "ba.json", report);
  fs::rename(temporary, output);
  std::cout << report.dump() << '\n';
}

void RunMap(const Arguments& args) {
  args.Validate({"--database", "--seed", "--images", "--output", "--workers",
                 "--blas-threads"});
  const auto output = args.Path("--output");
  if (fs::exists(output))
    throw std::runtime_error("Output already exists: " + output.u8string());
  const auto temporary =
      fs::path(output.native() + fs::path(".pending").native());
  if (fs::exists(temporary))
    throw std::runtime_error("Unfinished output exists: " +
                             temporary.u8string());
  fs::create_directories(temporary);
  fs::create_directories(temporary / "models");
  const auto database = temporary / "database.db";
  fs::copy_file(args.Path("--database"), database);
  auto options = std::make_shared<colmap::IncrementalPipelineOptions>();
  options->num_threads = args.Integer("--workers", 4);
  options->random_seed = 271828;
  options->multiple_models = false;
  options->min_model_size = 10;
  options->min_num_matches = 30;
  options->structure_less_registration_fallback = false;
  options->ba_refine_focal_length = false;
  options->ba_refine_principal_point = false;
  options->ba_refine_extra_params = false;
  options->ba_refine_sensor_from_rig = false;
  options->use_prior_position = false;
  options->mapper.abs_pose_refine_focal_length = false;
  options->mapper.abs_pose_refine_extra_params = false;
  options->mapper.abs_pose_max_error = 6;
  options->mapper.abs_pose_min_num_inliers = 30;
  options->mapper.filter_max_reproj_error = 4;
  options->mapper.filter_min_tri_angle = 1;
  options->triangulation.min_angle = 1;
  options->ba_local_max_num_iterations = 35;
  options->ba_global_max_num_iterations = 60;
  auto manager = std::make_shared<colmap::ReconstructionManager>();
  manager->Read(args.Path("--seed"));
  const auto started = Clock::now();
  if (!colmap::RunIncrementalMapperImpl(database, args.Path("--images"),
                                        temporary / "models", options,
                                        manager) ||
      manager->Size() == 0)
    throw std::runtime_error("Mapping did not reconstruct a component");
  auto best = manager->Get(0);
  for (size_t i = 1; i < manager->Size(); ++i)
    if (manager->Get(i)->NumRegImages() > best->NumRegImages())
      best = manager->Get(i);
  auto depth = Cheirality(*best);
  if (depth.at("invalid_depth").get<uint64_t>() != 0)
    throw std::runtime_error("Mapped reconstruction has invalid depths");
  fs::create_directories(temporary / "sparse/0");
  best->Write(temporary / "sparse/0");
  colmap::Reconstruction reread;
  reread.Read(temporary / "sparse/0");
  if (!reread.IsValid())
    throw std::runtime_error("Invalid serialized mapping result");
  Json report = {{"elapsed_seconds", Seconds(started)},
                 {"registered", best->NumRegImages()},
                 {"points", best->NumPoints3D()},
                 {"observations", best->ComputeNumObservations()},
                 {"cheirality", depth},
                 {"environment", Environment()}};
  report["resources"] = Resources();
  WriteJson(temporary / "mapping.json", report);
  fs::rename(temporary, output);
  std::cout << report.dump() << '\n';
}

int Main(const std::vector<std::string>& argv) {
  try {
    Arguments args(argv);
    openblas_set_num_threads(args.Integer("--blas-threads", 1));
    if (args.command == "info") {
      args.Validate({"--blas-threads"});
      std::cout << Environment().dump(2) << '\n';
    } else if (args.command == "ba")
      RunBA(args);
    else if (args.command == "map")
      RunMap(args);
    else if (args.command == "geometry")
      RunGeometry(args);
    else if (args.command == "sfm")
      RunVisualPipeline(args);
    else if (args.command == "filter-tracks")
      RunTrackFilter(args);
    else if (args.command == "quality")
      RunQuality(args);
    else if (args.command == "las-ply")
      RunLasPly(args);
    else if (args.command == "register")
      RunRegistration(args);
    else if (args.command == "export")
      RunExport(args);
    else if (args.command == "check")
      RunCheck(args);
    else if (args.command == "run")
      RunProduction(args);
    else if (args.command == "prepare") {
      args.Validate({"--input", "--work", "--blas-threads"});
      std::cout
          << PrepareScene(args.Path("--input"), args.Path("--work")).dump()
          << '\n';
    } else if (args.command == "help" || args.command == "--help")
      std::cout
          << "geosunpre-native info\n"
             "geosunpre-native run --input RAW_INPUT --work NEW_WORK [--stage "
             "prepare|masks|features|matches|observations|calibration|sfm|"
             "finalize|register|export|all]\n"
             "geosunpre-native run --work WORK\n"
             "geosunpre-native export --work WORK --output NEW_OUTPUT\n"
             "geosunpre-native check --output DATASET\n"
             "geosunpre-native ba --input MODEL --initial-model MODEL --output "
             "NEW_DIR [--stage fixed|focal_low_radial|all_radial|principal] "
             "[--protect-depth] [--gpu]\n"
             "geosunpre-native map --database DATABASE --seed MODEL --images "
             "IMAGES --output NEW_DIR\n";
    else
      throw std::invalid_argument("Unknown command: " + args.command);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "geosunpre-native: " << error.what() << '\n';
    return 1;
  }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) { return Main(Utf8Arguments(argc, argv)); }
#else
int main(int argc, char** argv) { return Main(Utf8Arguments(argc, argv)); }
#endif
