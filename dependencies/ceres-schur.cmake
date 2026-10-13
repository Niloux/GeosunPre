# Ceres 2.2 skips point back-substitution when explicit Schur CG reaches its
# iteration limit. Complete that usable inexact step, as implicit Schur does.
set(_ceres_schur_path "${ceres_SOURCE_DIR}/internal/ceres/schur_complement_solver.cc")
file(READ "${_ceres_schur_path}" _ceres_schur_source)
set(_ceres_schur_original [=[  if (summary.termination_type == LinearSolverTerminationType::SUCCESS) {
    eliminator_->BackSubstitute(]=])
set(_ceres_schur_fixed [=[  if (summary.termination_type == LinearSolverTerminationType::SUCCESS ||
      summary.termination_type == LinearSolverTerminationType::NO_CONVERGENCE) {
    eliminator_->BackSubstitute(]=])
string(FIND "${_ceres_schur_source}" "${_ceres_schur_fixed}" _ceres_schur_patched)
if(_ceres_schur_patched EQUAL -1)
  string(FIND "${_ceres_schur_source}" "${_ceres_schur_original}" _ceres_schur_match)
  if(_ceres_schur_match EQUAL -1)
    message(FATAL_ERROR "Cannot apply the Ceres Schur back-substitution fix: source changed")
  endif()
  string(REPLACE "${_ceres_schur_original}" "${_ceres_schur_fixed}"
    _ceres_schur_source "${_ceres_schur_source}")
  file(WRITE "${_ceres_schur_path}" "${_ceres_schur_source}")
endif()
