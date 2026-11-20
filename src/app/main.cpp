#include "app/pipeline.h"
#include "app/build_info.h"
#include "io/colmap.h"
#include "io/las.h"
#include "frontend/inference.h"
#include "frontend/frontend.h"
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
     "[--project FILE.pro] [--config JSON] [--no-las-supervision]\n"
     "  geosunpre run --work WORK [--stage "
     "prepare|masks|features|matches|sfm|align|shutter|cleanup|export]",
     RunCommand},
    {"prepare", "--input INPUT [--project FILE.pro] [--output OUT] [--work WORK] [--debug]", RunCommand},
    {"export", "--work WORK --output OUT [--no-ply] [--no-roi] [--ply-voxel METRES] [--no-las-supervision]",
     Export},
    {"check", "--output DIR [--project FILE.pro]",
     [](const Arguments& args) {
       args.Validate({"--output", "--project"});
       std::cout << CheckBundle(args.Path("--output"), args.Get("--project")).dump(2) << '\n';
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
  std::cout << "  Use --debug to keep the work directory for stage resume or export.\n"
               "  New runs use the most recently modified .pro by default.\n"
               "  Delivery options: [--supervision-source moge|las] [--supervision-models MODELS]\n"
               "    [--supervision-max-size PIXELS] [--supervision-boundary PIXELS]\n"
               "    [--supervision-mask-dir DIR] [--supervision-cache-mb MIB]\n"
               "    [--no-static-masks] [--no-crop-valid-domain] [--no-roi]\n"
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
