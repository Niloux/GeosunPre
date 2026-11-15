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
namespace {
struct Command {
  const char* name;
  const char* usage;
  void (*execute)(const Arguments&);
};

void RunCommand(const Arguments& args) {
  Log("GeosunPre: " + BuildInfo().dump());
  Run(args);
}

void Help(const Arguments&);

const Command Commands[] = {
    {"run",
     "--input INPUT [--output OUT] [--work WORK] [--debug] [--models MODELS] "
     "[--config JSON] [--no-las-supervision]\n"
     "  geosunpre run --work WORK [--stage "
     "prepare|masks|features|matches|sfm|align|shutter|cleanup|export]",
     RunCommand},
    {"prepare", "--input INPUT [--output OUT] [--work WORK] [--debug]", RunCommand},
    {"export", "--work WORK --output OUT [--no-ply] [--ply-voxel METRES] [--no-las-supervision]",
     Export},
    {"cleanup",
     "--work WORK --output OUT [--new-work NEW_WORK] [--debug] [--workers N] "
     "[--no-ply] [--ply-voxel METRES] [--no-las-supervision]",
     Postprocess},
    {"check", "--output DIR",
     [](const Arguments& args) {
       args.Validate({"--output"});
       std::cout << CheckBundle(args.Path("--output")).dump(2) << '\n';
     }},
    {"check-supervision", "--output DIR [--compare DIR]",
     [](const Arguments& args) {
       args.Validate({"--output", "--compare"});
       std::cout << MeasureSupervision(args.Path("--output"), args.Get("--compare").empty()
                                                                  ? fs::path()
                                                                  : args.Path("--compare"))
                        .dump(2)
                 << '\n';
     }},
    {"static-masks",
     "--output DELIVERY --input INPUT --las LAS --new-masks DIR [--enhanced-photos]",
     [](const Arguments& args) {
       args.Validate({"--output", "--input", "--las", "--new-masks", "--enhanced-photos"});
       std::cout << RefineStaticMasks(args.Path("--output"), args.Path("--input"),
                                      args.Path("--las"), args.Get("--enhanced-photos") == "true",
                                      args.Path("--new-masks"))
                        .dump(2)
                 << '\n';
     }},
    {"info", "",
     [](const Arguments& args) {
       args.Validate({});
       auto info = BuildInfo();
       info["runtime"] = "C++17/CUDA/TensorRT 10.13";
       info["gpu"] = GpuInfo();
       info["default_models"] = DefaultModels().u8string();
       std::cout << info.dump(2) << '\n';
     }},
    {"check-models", "--image FILE [--models MODELS]",
     [](const Arguments& args) {
       args.Validate({"--models", "--image"});
       std::cout << CheckModels(
                        args.Get("--models").empty() ? DefaultModels() : args.Path("--models"),
                        args.Path("--image"))
                        .dump(2)
                 << '\n';
     }},
    {"check-monocular",
     "--input INPUT --image NAME --output DIR [--models MODELS] [--supervision-max-size PIXELS]",
     [](const Arguments& args) {
       args.Validate({"--input", "--image", "--output", "--models", "--supervision-max-size"});
       std::cout << CheckMonocular(
                        args.Path("--input"),
                        args.Get("--models").empty() ? DefaultModels() : args.Path("--models"),
                        args.Get("--image"), args.Path("--output"),
                        args.Integer("--supervision-max-size", 1600))
                        .dump(2)
                 << '\n';
     }},
    {"las-ply", "--las FILE --output FILE [--voxel METRES]",
     [](const Arguments& args) {
       args.Validate({"--las", "--output", "--voxel"});
       std::cout << ExportLasPly(args.Path("--las"), ReadLasHeader(args.Path("--las")),
                                 args.Path("--output"), args.Number("--voxel", .05))
                        .dump(2)
                 << '\n';
     }},
    {"help", "", Help},
    {"--help", nullptr, Help},
};

void Help(const Arguments&) {
  std::cout << "GeosunPre " << BuildVersion << " - native dual-fisheye/LAS reconstruction\n";
  for (const auto& command : Commands) {
    if (command.usage)
      std::cout << "  geosunpre " << command.name << (*command.usage ? " " : "") << command.usage
                << '\n';
  }
  std::cout << "  Use --debug to keep the work directory for export or cleanup.\n"
               "  Delivery options: [--supervision-source moge|las] [--supervision-models MODELS]\n"
               "    [--supervision-max-size PIXELS] [--supervision-boundary PIXELS]\n"
               "    [--supervision-mask-dir DIR] [--supervision-cache-mb MIB]\n"
               "    [--no-static-masks] [--no-crop-valid-domain]\n"
               "  See README.md and docs/output.md for defaults and delivery details.\n";
}
}  // namespace

int Main(const std::vector<std::string>& argv) {
  try {
    cv::setNumThreads(1);
    const Arguments args(argv);
    for (const auto& command : Commands) {
      if (args.command == command.name) {
        command.execute(args);
        return 0;
      }
    }
    throw std::runtime_error("Unknown command: " + args.command);
  } catch (const std::exception& error) {
    std::cerr << "geosunpre: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace geosun
int wmain(int argc, wchar_t** argv) { return geosun::Main(geosun::Utf8Arguments(argc, argv)); }
