#include "app/pipeline.h"
#include "app/build_info.h"
#include "io/colmap.h"
#include "io/las.h"
#include "frontend/inference.h"
#include "frontend/frontend.h"
#include <opencv2/core.hpp>
#include <iostream>

namespace geosun {
int Main(const std::vector<std::string>& argv) {
  try {
    cv::setNumThreads(1);
    Arguments args(argv);
    if (args.command == "run" || args.command == "prepare") {
      Log("GeosunPre: " + BuildInfo().dump());
      Run(args);
    } else if (args.command == "export")
      Export(args);
    else if (args.command == "refine" || args.command == "cleanup")
      Postprocess(args);
    else if (args.command == "check") {
      args.Validate({"--output", "--work"});
      std::cout << CheckBundle(args.Get("--output").empty() ? args.Path("--work") / "colmap"
                                                            : args.Path("--output"))
                       .dump(2)
                << '\n';
    } else if (args.command == "info") {
      args.Validate({});
      auto info = BuildInfo();
      info["runtime"] = "C++17/CUDA/TensorRT 10.13";
      info["gpu"] = GpuInfo();
      info["default_models"] = DefaultModels().u8string();
      std::cout << info.dump(2) << '\n';
    } else if (args.command == "check-models") {
      args.Validate({"--models", "--image"});
      std::cout << CheckModels(
                       args.Get("--models").empty() ? DefaultModels() : args.Path("--models"),
                       args.Path("--image"))
                       .dump(2)
                << '\n';
    } else if (args.command == "las-ply") {
      args.Validate({"--las", "--output", "--voxel"});
      std::cout << ExportLasPly(args.Path("--las"), args.Path("--output"),
                                args.Number("--voxel", .05))
                       .dump(2)
                << '\n';
    } else if (args.command == "help" || args.command == "--help") {
      std::cout
          << "GeosunPre " << BuildVersion
          << " - native dual-fisheye/LAS reconstruction\n"
             "  geosunpre run --input INPUT --work WORK [--models MODELS] [--config JSON] "
             "[--refine] [--no-las-supervision]\n"
             "  geosunpre run --work WORK [--stage "
             "prepare|masks|features|matches|sfm|align|refine|shutter|cleanup|export]\n"
             "    refine is disabled by default; enable with --refine or config refine=true\n"
             "    shutter (rolling-shutter correction) runs by default; config "
             "rolling_shutter=false removes it\n"
             "  geosunpre prepare --input INPUT --work WORK [--refine]\n"
             "  geosunpre export --work WORK [--output DIR] [--no-ply] [--ply-voxel METRES] "
             "[--no-las-supervision]\n"
             "  geosunpre refine --work WORK --output NEW_WORK [--workers N] [--no-ply] "
             "[--ply-voxel METRES] [--no-las-supervision]\n"
             "  geosunpre cleanup --work WORK --output NEW_WORK [--workers N] [--no-ply] "
             "[--ply-voxel METRES] [--no-las-supervision]\n"
             "    LAS depth/normal supervision is enabled for run/export/refine/cleanup\n"
             "    [--supervision-max-size PIXELS] [--supervision-boundary PIXELS] "
             "[--supervision-mask-dir DIR] [--supervision-cache-mb MIB]\n"
             "    LAS coordinate GPU cache limit defaults to 1024 MiB (1-4096)\n"
             "    [--rolling-shutter-photos rectified|original] for corrected models: photos "
             "resampled to the reference poses (default), or the original photos with "
             "rolling_shutter.json\n"
             "    --las-supervision explicitly enables it; conflicts with --no-las-supervision\n"
             "  geosunpre check --output DIR\n"
             "  geosunpre info\n"
             "  geosunpre check-models --image FILE [--models MODELS]\n"
             "  geosunpre las-ply --las FILE --output FILE [--voxel METRES]\n";
    } else
      throw std::runtime_error("Unknown command: " + args.command);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "geosunpre: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace geosun
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) { return geosun::Main(geosun::Utf8Arguments(argc, argv)); }
#else
int main(int argc, char** argv) { return geosun::Main(geosun::Utf8Arguments(argc, argv)); }
#endif
