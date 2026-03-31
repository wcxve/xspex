#ifndef XSPEX_XSPEC_HPP_
#define XSPEX_XSPEC_HPP_

#include <cstdlib>
#include <fstream>
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
#include "utils.hpp"

namespace xspex::xspec
{
struct DatabaseVersionSpec {
    const char* setting_key;
};

inline constexpr DatabaseVersionSpec atomdb_version_spec{"ATOMDB_VERSION"};

inline constexpr DatabaseVersionSpec spex_version_spec{"SPEX_VERSION"};

inline constexpr DatabaseVersionSpec nei_version_spec{"NEI_VERSION"};

inline std::string join_path(std::string base, const std::string& name)
{
    if (!base.empty() && base.back() != '/') {
        base.push_back('/');
    }
    return base + name;
}

inline std::string default_database_version(const DatabaseVersionSpec& spec)
{
    const auto settings = XSutility::readSettingsFile(
        join_path(FunctionUtility::managerPath(), "Xspec.init"));
    const auto it = settings.find(spec.setting_key);
    if (it != settings.end() && !it->second.empty() &&
        utils::to_lower_case(it->second) != "latest") {
        return it->second;
    }
    return {};
}

inline std::string resolve_database_version(const std::string& version,
                                            const DatabaseVersionSpec& spec)
{
    if (utils::to_lower_case(version) != "latest") {
        return version;
    }

    const auto latest_path =
        join_path(FunctionUtility::modelDataPath(), "latest.txt");
    std::ifstream latest_file(latest_path);
    if (!latest_file.is_open()) {
        const auto default_version = default_database_version(spec);
        if (!default_version.empty()) {
            return default_version;
        }
        std::ostringstream oss;
        oss << "failed to resolve latest value for " << spec.setting_key
            << ": \"" << latest_path
            << "\" is missing and Xspec.init does not provide a concrete "
               "version";
        throw std::runtime_error(oss.str());
    }

    const auto expected_key = std::string(spec.setting_key) + ":";
    std::string key;
    std::string resolved;
    while (latest_file >> key >> resolved) {
        if (key == expected_key) {
            return resolved;
        }
    }

    std::ostringstream oss;
    oss << "failed to resolve latest value for " << spec.setting_key
        << " from \"" << latest_path << "\"";
    throw std::runtime_error(oss.str());
}

inline void normalize_database_versions()
{
    const auto atomdb_version = resolve_database_version(
        FunctionUtility::atomdbVersion(), atomdb_version_spec);
    if (atomdb_version != FunctionUtility::atomdbVersion()) {
        FunctionUtility::atomdbVersion(atomdb_version);
    }

    const auto spex_version = resolve_database_version(
        FunctionUtility::spexVersion(), spex_version_spec);
    if (spex_version != FunctionUtility::spexVersion()) {
        FunctionUtility::spexVersion(spex_version);
    }

    const auto nei_version = resolve_database_version(
        FunctionUtility::neiVersion(), nei_version_spec);
    if (nei_version != FunctionUtility::neiVersion()) {
        FunctionUtility::neiVersion(nei_version);
    }
}

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
                 "default to H0=70.0, q0=0.0, lambda0=0.73\n";
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
        // Resolve "latest" database version aliases for older XSPEC builds
        // where FNINIT stores the keyword verbatim in the model string DB.
        normalize_database_versions();
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
