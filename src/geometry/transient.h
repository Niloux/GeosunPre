#pragma once
#include "io/scene.h"

namespace geosun {
// LAS returns from what moved during the scan (people walking past, opened doors): the
// scanner's own rays at other times pass through the space they occupied. drop holds one flag
// per LAS record and is empty when the LAS has no GPS time; report goes to the delivery.
struct TransientLas {
  std::vector<bool> drop;
  Json report;
};
// The filter's parameters, part of the delivery request.
Json TransientLasRequest();
TransientLas FindTransientLas(const Scene& scene);
}  // namespace geosun
