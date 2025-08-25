#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>

#include "interface.hpp"
#include "wrapper.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(libxspex, m)
{
    m.def("xla_ffi_handlers",
          &xspex::wrapper::xla_ffi_handlers,
          "Get XLA FFI handlers for XSPEC model functions.");

    m.def("chatter",
          &xspex::interface::get_chatter,
          "Get XSPEC console chatter level.");

    m.def("chatter",
          &xspex::interface::set_chatter,
          "level"_a,
          "Set XSPEC console chatter level.");

    m.def("abund",
          &xspex::interface::get_abund,
          "Get abundance table used in XSPEC.");

    m.def("abund",
          &xspex::interface::set_abund,
          "table"_a,
          "Set abundance table used in XSPEC.");

    m.def("abund_file",
          &xspex::interface::set_abund_file,
          "file"_a,
          "Set abundance file used in XSPEC.");

    m.def("xsect",
          &xspex::interface::get_xsect,
          "Get photo-electric cross-section table used in XSPEC.");

    m.def("xsect",
          &xspex::interface::set_xsect,
          "table"_a,
          "Set photo-electric cross-section table used in XSPEC.");

    m.def("cosmo",
          &xspex::interface::get_cosmo,
          "Get cosmological parameters used in XSPEC.");

    m.def("cosmo",
          &xspex::interface::set_cosmo,
          "H0"_a,
          "q0"_a,
          "lambda0"_a,
          "Set cosmological parameters used in XSPEC.");

    m.def("xspec_version",
          &xspex::interface::xspec_version,
          "Get XSPEC version.");

    m.def("mstr",
          &xspex::interface::get_mstrs,
          "Get all model string used in XSPEC.");

    m.def("mstr",
          &xspex::interface::get_mstr,
          "key"_a,
          "Get model string for a given key.");

    m.def("mstr",
          &xspex::interface::set_mstr,
          "key"_a,
          "value"_a,
          "Set model string for a given key.");

    m.def("mstr",
          &xspex::interface::set_mstrs,
          "dic"_a,
          "Set multiple model strings.");

    m.def("clear_mstr",
          &xspex::interface::clear_mstr,
          "Clear all model strings in XSPEC.");

    m.def("xflt", &xspex::interface::get_xflts, "Get all XFLT entries.");

    m.def("xflt",
          &xspex::interface::get_xflt,
          "spec_num"_a,
          "Get XFLT entries for a given spectrum number.");

    m.def("xflt",
          &xspex::interface::set_xflt,
          "spec_num"_a,
          "dic"_a,
          "Set XFLT entries for a given spectrum number.");

    m.def("xflt",
          &xspex::interface::set_xflts,
          "maps"_a,
          "Set multiple XFLT entries for multiple spectra.");

    m.def("clear_xflt",
          &xspex::interface::clear_xflt,
          "spec_num"_a,
          "Clear XFLT entries for a given spectrum number.");

    m.def("clear_xflt",
          &xspex::interface::clear_all_xflt,
          "Clear all XFLT entries.");

    m.def("xflt_sync_to_xspec",
          &xspex::interface::xflt_sync_to_xspec,
          "Sync XFLT entries to XSPEC in current process.");

    m.def("clear_xflt_xspec",
          &xspex::interface::clear_xflt_xspec,
          "Clear all XFLT entries of XSPEC in current process.");
}
