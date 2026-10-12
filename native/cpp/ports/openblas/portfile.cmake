vcpkg_download_distfile(ARCHIVE
    URLS "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.33/OpenBLAS-0.3.33-x64.zip"
    FILENAME "OpenBLAS-0.3.33-x64.zip"
    SHA512 ac50c79d0751f7e47c3d942114a946ef48b83289313573df5619557f93953801ad43efb49be95e955c4493edba529c25f9a592dd11aecae80454b29aacd85627)
vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}" NO_REMOVE_ONE_LEVEL)
file(COPY "${SOURCE_PATH}/include/" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/bin/libopenblas.dll" DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
file(INSTALL "${SOURCE_PATH}/lib/libopenblas.lib" DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${SOURCE_PATH}/bin/libopenblas.dll" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/bin")
file(INSTALL "${SOURCE_PATH}/lib/libopenblas.lib" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
file(INSTALL "${CURRENT_PORT_DIR}/OpenBLASConfig.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/openblas")
file(INSTALL "${CURRENT_PORT_DIR}/copyright" DESTINATION "${CURRENT_PACKAGES_DIR}/share/openblas")
