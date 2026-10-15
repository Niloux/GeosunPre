execute_process(COMMAND git describe --always --dirty WORKING_DIRECTORY "${SOURCE_ROOT}"
  OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT revision)
  set(revision "source-archive")
endif()
string(TIMESTAMP built_utc "%Y-%m-%dT%H:%M:%SZ" UTC)
get_filename_component(output_directory "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
file(WRITE "${OUTPUT}" "#include \"app/build_info.h\"\nnamespace geosun {\nconst char BuildVersion[] = \"${VERSION}\";\nconst char BuildRevision[] = \"${revision}\";\nconst char BuildTime[] = \"${built_utc}\";\n}\n")
