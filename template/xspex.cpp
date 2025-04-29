#include "xspex.hpp"

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

using namespace pybind11::literals;

py::dict xla_registrations() {
    py::dict reg;
    // Add the models, auto-generated from the model.dat file.

    // float
/*XLA_FLOAT_MODELS*/

    // double
/*XLA_DOUBLE_MODELS*/

    return reg;
}

PYBIND11_MODULE(_compiled, m) {
#ifdef XSPEX_VERSION
    m.attr("__version__") = STRINGIFY(XSPEX_VERSION);
#else
    m.attr("__version__") = "dev";
#endif
    m.doc() = R"doc(
Call Xspec models from Python
=============================

Part of codes are adapted from xspec-models-cxc, which is highly experimental.

The Xspec model library is automatically initialized on module loading.

Support routines
----------------
get_version - The version of the Xspec model library.
chatter - Get or set the Xspec chatter level.
abundance - Get or set the abundance-table setting.
cross_section - Get or set the cross-section-table setting.
element_abundance - Get the abundance for an element by name or atomic number.
element_name - Get the name of an element given the atomic number.
cosmology - Get or set the cosmology (H0, q0, lambda0) settings.

Table Models
------------
tableModel

Additive models
---------------
/*ADDMODELS*/

Multiplicative models
---------------------
/*MULMODELS*/

Convolution models
------------------
/*CONMODELS*/

)doc";
    m.def("_init", &xspex::init, "Initializes data directory locations needed by the models.");
    m.def("version", &xspex::get_version, "The version of the Xspec model library.");
    m.def("chatter", &xspex::get_chatter, "Get the Xspec chatter level.");
    m.def("chatter", &xspex::set_chatter, "Set the Xspec chatter level.", "lvl"_a);
    m.def("abundance", &xspex::get_abundance, "Get the abundance-table setting.");
    m.def("abundance", &xspex::set_abundance, "Set the abundance-table setting.", "table"_a);
    m.def("element_abundance", &xspex::abundance_by_name, "Get the abundance setting for an element given the name.", "name"_a);
    m.def("element_abundance", &xspex::abundance_by_z, "Get the abundance setting for an element given the atomic number.", "z"_a);
    m.def("element_name", &xspex::element_name_by_z, "Get the name of an element given the atomic number.", "z"_a);
    m.attr("number_elements") = xspex::number_elements;
    m.def("cross_section", &xspex::get_cross_section, "Get the cross-section-table setting.");
    m.def("cross_section", &xspex::set_cross_section, "Set the cross-section-table setting.", "table"_a);
    m.def("cosmology", &xspex::get_cosmology, "Get the current cosmology (H0, q0, lambda0).");
    m.def("cosmology", &xspex::set_cosmology, "Set the current cosmology (H0, q0, lambda0).", "H0"_a, "q0"_a, "lambda0"_a);

    // XFLT keyword handling: the names are hardly instructive. We could
    // just have an overloaded XFLT method which either queries or sets
    // the values, and then leave the rest to the user to do in Python.
    //
    m.def(
        "clear_xflt",
	    []() { return FunctionUtility::clearXFLT(); },
	    "Clear the XFLT database for all spectra."
    );

    m.def(
        "get_number_xflt",
	    [](int ifl) { return FunctionUtility::getNumberXFLT(ifl); },
	    "How many XFLT keywords are defined for the spectrum?",
	    "spectrum"_a=1
    );

    m.def(
        "get_xflt",
	    [](int ifl) { return FunctionUtility::getAllXFLT(ifl); },
	    "What are all the XFLT keywords for the spectrum?",
	    "spectrum"_a=1,
	    py::return_value_policy::reference
    );

    m.def(
        "get_xflt",
	    [](int ifl, int i) { return FunctionUtility::getXFLT(ifl, i); },
	    "Return the given XFLT key.",
	    "spectrum"_a, "key"_a
    );

    m.def(
        "get_xflt",
	    [](int ifl, string skey) { return FunctionUtility::getXFLT(ifl, skey); },
	    "Return the given XFLT name.",
	    "spectrum"_a, "name"_a
    );

    m.def(
        "in_xflt",
	    [](int ifl, int i) { return FunctionUtility::inXFLT(ifl, i); },
	    "Is the given XFLT key set?",
	    "spectrum"_a, "key"_a
    );

    m.def(
        "in_xflt",
	    [](int ifl, string skey) { return FunctionUtility::inXFLT(ifl, skey); },
	    "Is the given XFLT name set?.",
	    "spectrum"_a, "name"_a
    );

    m.def(
        "set_xflt",
	    [](int ifl, const std::map<string, Real>& values) { FunctionUtility::loadXFLT(ifl, values); },
	    "Set the XFLT keywords for a spectrum",
	    "spectrum"_a, "values"_a
    );

    // Model database - as with XFLT how much do we just leave to Python?
    //
    // What are the memory requirements?
    //
    m.def(
        "clear_model_string",
	    []() { return FunctionUtility::eraseModelStringDataBase(); },
	    "Clear the model string database."
    );

    m.def(
        "get_model_string",
	    []() { return FunctionUtility::modelStringDataBase(); },
	    "Get the model string database.",
	    py::return_value_policy::reference
    );

    m.def(
        "get_model_string",
	    [](const string& key) {
	        auto answer = FunctionUtility::getModelString(key);
	        if (answer == FunctionUtility::NOT_A_KEY()) throw pybind11::key_error(key);
	        return answer;
	    },
	    "Get the key from the model string database.",
	    "key"_a
	);

    m.def(
        "set_model_string",
	    [](const string& key, const string& value) { FunctionUtility::setModelString(key, value); },
	    "Get the key from the model string database.",
	    "key"_a, "value"_a
    );

    // "keyword" database values - similar to XFLT we could leave most of this to
    // Python.
    //
    m.def(
        "clear_db",
	    []() { return FunctionUtility::clearDb(); },
	    "Clear the keyword database."
    );

    m.def(
        "get_db",
	    []() { return FunctionUtility::getAllDbValues(); },
	    "Get the keyword database.",
	    py::return_value_policy::reference
    );

    // If the keyword is not an element then we get a string message and a set
    // return value. Catching this is annoying.
    //
    m.def(
        "get_db",
	    [](const string keyword) {
	        std::ostringstream local;
	        auto cerr_buff = std::cerr.rdbuf();
	        std::cerr.rdbuf(local.rdbuf());

	        // Assume this can not throw an error
	        auto answer = FunctionUtility::getDbValue(keyword);

	        std::cerr.rdbuf(cerr_buff);
	        if (answer == BADVAL) throw pybind11::key_error(keyword);

	        return answer;
	    },
	    "Get the keyword value from the database.",
	    "keyword"_a
    );

    m.def(
        "set_db",
	    [](const string keyword, const double value) {
	        FunctionUtility::loadDbValue(keyword, value);
	    },
	    "Set the keyword in the database to the given value.",
	    "keyword"_a, "value"_a
    );

    m.def("table_model", &xspex::wrapper_table_model<float>, "Call Xspec table model.", "table"_a, "table_type"_a, "pars"_a, "energies"_a, "spectrum"_a=1);

    m.def("xla_registrations", &xla_registrations, "Registrations of XLA ops.");

    // Add the models, auto-generated from the model.dat file.
/*MODELS*/
}
