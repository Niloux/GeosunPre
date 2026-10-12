#pragma once
#include "runtime.h"
void RunExport(const Arguments& args);
void RunCheck(const Arguments& args);
Json CheckExport(const fs::path& output);
