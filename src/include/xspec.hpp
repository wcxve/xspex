#ifndef XSPEX_XSPEC_HPP_
#define XSPEX_XSPEC_HPP_

#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

// clang-format off
#include <XSFunctions/Utilities/FunctionUtility.h>
#include <XSFunctions/Utilities/xsFortran.h>
#include <XSFunctions/funcWrappers.h>
#include <XSUtil/Utils/XSutility.h>
// clang-format on

#include "mdef.hpp"

namespace xspex::xspec
{
// Initialize the Cosmology settings from Xspec.init file.
inline void initialize_xspec_cosmology()
{
    std::map<std::string, std::string>::iterator it;
    std::istringstream cosmo;
    cosmo.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    double q{0}, h{0}, l{0};

    // Read the user's settings
    std::string user_path;
    std::map<std::string, std::string> user_settings;
    const char* home_env = getenv("HOME");
    if (home_env && *home_env) {
        user_path = std::string(home_env) + "/.xspec/Xspec.init";
        user_settings = XSutility::readSettingsFile(user_path);
    }
    it = user_settings.find("COSMO");
    if (it != user_settings.end()) {
        try {
            cosmo.clear();
            cosmo.str(it->second);
            cosmo >> h >> q >> l;
            FunctionUtility::setFunctionCosmoParams(h, q, l);
            return;
        } catch (...) {
            std::cerr << "Cosmology settings in initialization file "
                      << "corrupted - check " << user_path << "\n";
        }
    }

    // Read the default settings
    const auto& manager_loc = FunctionUtility::managerPath();
    const std::string default_path = manager_loc + "/Xspec.init";
    std::map<std::string, std::string> defaultSettings =
        XSutility::readSettingsFile(default_path);
    it = defaultSettings.find("COSMO");
    if (it != defaultSettings.end()) {
        try {
            cosmo.clear();
            cosmo.str(it->second);
            cosmo >> h >> q >> l;
            FunctionUtility::setFunctionCosmoParams(h, q, l);
            return;
        } catch (...) {
            std::cerr << "Cosmology settings in initialization file corrupted "
                      << "- check " << default_path << "\n";
        }
    }

    std::cerr << "Faild to get cosmology settings from initialization file, "
                 "default to H0=70.0, q0=0.0, lamba0=0.73\n";
    FunctionUtility::setFunctionCosmoParams(70.0, 0.0, 0.73);
}

// Initialize XSPEC model library. Must be called before calling any other
// XSPEC functions.
inline void initialize_xspec_model_library()
{
    // Check if HEADAS env is set
    if (!getenv("HEADAS")) {
        throw std::runtime_error("environment variable HEADAS was not found");
    }

    // Hide stdout for FNINIT, and restore it later
    std::ostringstream s;
    auto cout_buff = std::cout.rdbuf();
    std::cout.rdbuf(s.rdbuf());  // Hide stdout
    try {
        // Initialize XSPEC model library
        FNINIT();
        // Initialize the Cosmology settings
        initialize_xspec_cosmology();
        // Clear all XFLT values
        FunctionUtility::clearXFLT();
        // Restore stdout
        std::cout.rdbuf(cout_buff);
    } catch (...) {
        std::cout.rdbuf(cout_buff);
        std::cout << s.str();
        throw;
    }
}

using XSPEC_C_Func = void (*)(const double* energy,
                              int nflux,
                              const double* params,
                              int spec_num,
                              double* flux,
                              double* flux_error,
                              const char* init_string);

// Auto-generate function array from model definitions in mdef.hpp
inline XSPEC_C_Func functions[XSPEC_MODEL_COUNT]{
#define ENTRY(name, n_params) C_##name,
    XSPEC_MODELS XSPEC_CON_MODELS
#undef ENTRY
};
}  // namespace xspex::xspec

#endif  // XSPEX_XSPEC_HPP_
