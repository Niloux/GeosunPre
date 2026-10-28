#include "app/pipeline.h"
#include "app/build_info.h"
#include "io/colmap.h"
#include "io/las.h"
#include "frontend/inference.h"
#include "frontend/frontend.h"
#include "geometry/monocular.h"
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
      args.Validate({"--output"});
      std::cout << CheckBundle(args.Path("--output")).dump(2) << '\n';
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
    } else if (args.command == "check-monocular") {
      args.Validate({"--input", "--image", "--output", "--models", "--supervision-max-size"});
      std::cout << CheckMonocular(
                       args.Path("--input"),
                       args.Get("--models").empty() ? DefaultModels() : args.Path("--models"),
                       args.Get("--image"), args.Path("--output"),
                       args.Integer("--supervision-max-size", 1600))
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
             "  geosunpre run --input INPUT [--output OUT] [--work WORK] [--debug] "
             "[--models MODELS] [--config JSON] [--refine] [--no-las-supervision]\n"
             "    the delivery is written directly into OUT (default output/<input name>)\n"
             "    production (default): work and cache live in OUT.work and are removed once "
             "the delivery is published; an interrupted run resumes with the same command\n"
             "    --debug keeps the work directory (default work/<input name>) and its cache\n"
             "  geosunpre run --work WORK [--stage "
             "prepare|masks|features|matches|sfm|align|refine|shutter|cleanup|export]\n"
             "    refine is disabled by default; enable with --refine or config refine=true\n"
             "    shutter (rolling-shutter correction) runs by default; config "
             "rolling_shutter=false removes it\n"
             "    original Image/L and Image/R photos are used; config enhanced_photos=true "
             "prefers Image/enhanced\n"
             "  geosunpre prepare --input INPUT [--output OUT] [--work WORK] [--debug] "
             "[--refine]\n"
             "  geosunpre export --work WORK --output OUT [--no-ply] [--ply-voxel METRES] "
             "[--no-las-supervision]\n"
             "  geosunpre refine --work WORK --output OUT [--new-work NEW_WORK] [--debug] "
             "[--workers N] [--no-ply] [--ply-voxel METRES] [--no-las-supervision]\n"
             "  geosunpre cleanup --work WORK --output OUT [--new-work NEW_WORK] [--debug] "
             "[--workers N] [--no-ply] [--ply-voxel METRES] [--no-las-supervision]\n"
             "    export, refine and cleanup read the cache of a --debug work directory\n"
             "    LAS depth/normal supervision is enabled for run/export/refine/cleanup\n"
             "    [--supervision-max-size PIXELS] [--supervision-boundary PIXELS] "
             "[--supervision-mask-dir DIR] [--supervision-cache-mb MIB]\n"
             "    --supervision-source las|moge: LAS-projected maps (default) or MoGe-2 maps "
             "scaled per face to the LAS; moge uses the geometry engine of "
             "--supervision-models MODELS (default models)\n"
             "    LAS coordinate GPU cache limit defaults to 1024 MiB (1-4096)\n"
             "    --las-supervision explicitly enables it; conflicts with --no-las-supervision\n"
             "    rolling-shutter corrected models are delivered with the original photos and "
             "rolling_shutter.json\n"
             "    photos, masks and the model are cut to each lens's valid domain, losslessly on "
             "the JPEG MCU grid; --no-crop-valid-domain delivers full frames\n"
             "    --crop-valid-domain explicitly enables it; conflicts with --no-crop-valid-domain\n"
             "  geosunpre check --output DIR\n"
             "  geosunpre info\n"
             "  geosunpre check-models --image FILE [--models MODELS]\n"
             "  geosunpre check-monocular --input INPUT --image NAME --output DIR "
             "[--models MODELS] [--supervision-max-size PIXELS]\n"
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
