#pragma once
#include <colmap/scene/reconstruction.h>

#include <stdexcept>

#include "runtime.h"
class VisualAcceptanceRejected : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
Json VisualQuality(const colmap::Reconstruction& rec, const Json& scene,
                   const Json& settings = Json::object());
Json SolveBA(colmap::Reconstruction& rec, const Arguments& args,
             const Json& initial);
Json Cheirality(const colmap::Reconstruction& rec);
void RunMap(const Arguments& args);
void RunGeometry(const Arguments& args);
void RunVisualPipeline(const Arguments& args);
void RunTrackFilter(const Arguments& args);
void RunQuality(const Arguments& args);
